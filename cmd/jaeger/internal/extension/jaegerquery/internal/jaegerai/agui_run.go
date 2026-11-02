// Copyright (c) 2026 The Jaeger Authors.
// SPDX-License-Identifier: Apache-2.0

package jaegerai

import (
	"context"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	aguievents "github.com/ag-ui-protocol/ag-ui/sdks/community/go/pkg/core/events"
	aguisse "github.com/ag-ui-protocol/ag-ui/sdks/community/go/pkg/encoding/sse"
	"github.com/coder/acp-go-sdk"
)

// aguiRunIDSeq is the process-wide monotonic counter that newIDStem
// appends to its time-derived stem.
var aguiRunIDSeq atomic.Uint64

// aguiRun is the server side of one AG-UI run for one chat turn. It writes the
// run's events as AG-UI SSE frames into the HTTP response; acpClient feeds it
// the agent's output and the turn-scoped MCP endpoint feeds it UI tool calls.
//
// Events are constructed via the typed AG-UI SDK constructors so that
// schema-required fields (toolCallName, messageId, etc.) are positional —
// forgetting one is a compile error rather than a runtime ZodError on the
// frontend. Framing is delegated to the SDK's SSEWriter.
//
// All mutable fields (closed, threadID, runID, messageID, textOpen) are
// guarded by mu. acpClient feeds agent output on a goroutine other
// than the one driving Prompt, and lifecycle calls (startRun, finishRun,
// failRun) come from the chat handler, so every entry point that touches
// state or writes a frame must acquire the lock first.
type aguiRun struct {
	requestCtx context.Context
	w          http.ResponseWriter
	sse        *aguisse.SSEWriter
	mu         sync.Mutex
	closed     bool
	threadID   string
	runID      string
	messageID  string
	textOpen   bool
}

func newAGUIRun(ctx context.Context, w http.ResponseWriter, threadID, runID string) *aguiRun {
	return &aguiRun{
		requestCtx: ctx,
		w:          w,
		sse:        aguisse.NewSSEWriter(),
		threadID:   threadID,
		runID:      runID,
	}
}

// emit writes a typed AG-UI event as a single SSE frame. The SDK's
// SSEWriter handles JSON encoding, newline escaping, and flushing.
// Must be called with c.mu held; on context cancellation or write error
// the run is marked closed so subsequent emissions are dropped.
func (c *aguiRun) emit(event aguievents.Event) {
	if c.closed {
		return
	}
	if c.requestCtx != nil {
		select {
		case <-c.requestCtx.Done():
			c.closed = true
			return
		default:
		}
	}
	if err := c.sse.WriteEvent(c.requestCtx, c.w, event); err != nil {
		c.closed = true
	}
}

// startRun emits RUN_STARTED and allocates a message id for the assistant
// text stream. ThreadID and RunID are preserved if the caller provided them
// via the AG-UI RunAgentInput payload; otherwise time-derived defaults fill
// in so the frontend always sees stable identifiers per run.
func (c *aguiRun) startRun() {
	c.mu.Lock()
	defer c.mu.Unlock()
	// Allocate one stem per startRun and derive all three IDs from it.
	// Sharing the stem makes the trio visibly correlated in logs/traces — a
	// thread-N-K, run-N-K, msg-N-K triplet tells a debugger at a glance
	// they came from the same run startup. Only the first text message of
	// the run carries the shared stem; closeText clears messageID and
	// ensureTextStart mints a fresh stem for each later message.
	stem := newIDStem()
	if c.threadID == "" {
		c.threadID = "thread-" + stem
	}
	if c.runID == "" {
		c.runID = "run-" + stem
	}
	if c.messageID == "" {
		c.messageID = "msg-" + stem
	}
	c.emit(aguievents.NewRunStartedEvent(c.threadID, c.runID))
}

// newIDStem returns "<nanos>-<seq>". The nanosecond keeps IDs time-sortable
// across processes, and the counter keeps two stems minted in the same
// nanosecond distinct on systems with coarse clock resolution.
func newIDStem() string {
	return fmt.Sprintf("%d-%d", time.Now().UnixNano(), aguiRunIDSeq.Add(1))
}

// finishRun closes any open text message and emits RUN_FINISHED. The
// AG-UI schema has no top-level stopReason field, so the sidecar's stop
// reason is forwarded via the schema-supported `result` payload as
// `{"stopReason": "<reason>"}`. Empty stop reasons are omitted so the
// emitted event stays compact.
func (c *aguiRun) finishRun(stopReason string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.closeText()
	if stopReason != "" {
		c.emit(aguievents.NewRunFinishedEventWithOptions(
			c.threadID, c.runID,
			aguievents.WithResult(map[string]any{"stopReason": stopReason}),
		))
		return
	}
	c.emit(aguievents.NewRunFinishedEvent(c.threadID, c.runID))
}

// failRun emits RUN_ERROR with the supplied message. It is safe to call from
// any handler error path; an open text message is closed first so frontends
// can finalize rendering before reporting the failure.
func (c *aguiRun) failRun(message string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.closeText()
	c.emit(aguievents.NewRunErrorEvent(message, aguievents.WithRunID(c.runID)))
}

// ensureTextStart emits TEXT_MESSAGE_START when streamed assistant text is
// observed and no message is open. Subsequent chunks are emitted as
// TEXT_MESSAGE_CONTENT. A messageID is allocated lazily here as well as in
// startRun so an out-of-order text chunk (no preceding startRun) still
// produces a schema-valid event — the SDK rejects events with empty required
// fields at encode time. Must be called with c.mu held.
func (c *aguiRun) ensureTextStart() {
	if c.textOpen {
		return
	}
	if c.messageID == "" {
		c.messageID = "msg-" + newIDStem()
	}
	c.emit(aguievents.NewTextMessageStartEvent(c.messageID))
	c.textOpen = true
}

// closeText emits TEXT_MESSAGE_END for the open text message, if any, and
// clears messageID so the next text chunk starts a new message. The
// @assistant-ui/react-ag-ui aggregator keeps one text part per messageId, so
// text that arrives after a tool call must carry a new id to render after
// the tool call rather than be appended to the text before it. Must be
// called with c.mu held.
func (c *aguiRun) closeText() {
	if !c.textOpen {
		return
	}
	c.emit(aguievents.NewTextMessageEndEvent(c.messageID))
	c.textOpen = false
	c.messageID = ""
}

// appendText streams one chunk of assistant text as TEXT_MESSAGE_CONTENT,
// opening a text message first when none is open. The AG-UI SDK rejects an
// empty delta and closes the stream, so an empty chunk is dropped. Whitespace
// between tokens is a real delta and is kept, but whitespace before the first
// visible text must not open a message.
func (c *aguiRun) appendText(text string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if text == "" || (!c.textOpen && strings.TrimSpace(text) == "") {
		return
	}
	c.ensureTextStart()
	c.emit(aguievents.NewTextMessageContentEvent(c.messageID, text))
}

// startToolCall closes any open text message and emits TOOL_CALL_START for a
// tool the agent invoked, followed by TOOL_CALL_ARGS when rawInput is known.
func (c *aguiRun) startToolCall(toolCallID, toolName string, rawInput any) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.closeText()
	c.emit(aguievents.NewToolCallStartEvent(toolCallID, toolName))
	if rawInput != nil {
		c.emit(aguievents.NewToolCallArgsEvent(toolCallID, marshalToolArgsDelta(rawInput)))
	}
}

// updateToolCall emits the later events of a tool call: TOOL_CALL_ARGS when
// rawInput is present, TOOL_CALL_RESULT when rawOutput is present, and
// TOOL_CALL_END when the call has finished.
func (c *aguiRun) updateToolCall(toolCallID string, rawInput, rawOutput any, finished bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if rawInput != nil {
		c.emit(aguievents.NewToolCallArgsEvent(toolCallID, marshalToolArgsDelta(rawInput)))
	}
	if rawOutput != nil {
		c.emit(aguievents.NewToolCallResultEvent(
			toolResultMessageID(acp.ToolCallId(toolCallID)),
			toolCallID,
			flattenToolResultContent(rawOutput),
		))
	}
	if finished {
		c.emit(aguievents.NewToolCallEndEvent(toolCallID))
	}
}

// EmitUIToolCall fires the TOOL_CALL_START / TOOL_CALL_ARGS /
// TOOL_CALL_END lifecycle for a UI tool the agent invoked via the
// turn-scoped MCP endpoint, writing the events onto the browser SSE stream so
// the frontend executes the tool. It acquires c.mu like the other entry points,
// so it is safe to call concurrently with the acpClient path.
func (c *aguiRun) EmitUIToolCall(toolCallID, toolName string, rawArgs any) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.closeText()
	c.emit(aguievents.NewToolCallStartEvent(toolCallID, toolName))
	if rawArgs != nil {
		c.emit(aguievents.NewToolCallArgsEvent(toolCallID, marshalToolArgsDelta(rawArgs)))
	}
	c.emit(aguievents.NewToolCallEndEvent(toolCallID))
}
