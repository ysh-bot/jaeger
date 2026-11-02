// Copyright (c) 2026 The Jaeger Authors.
// SPDX-License-Identifier: Apache-2.0

package jaegerai

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	aguievents "github.com/ag-ui-protocol/ag-ui/sdks/community/go/pkg/core/events"
	aguisse "github.com/ag-ui-protocol/ag-ui/sdks/community/go/pkg/encoding/sse"
	"github.com/coder/acp-go-sdk"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type errResponseWriter struct {
	header http.Header
}

func (w *errResponseWriter) Header() http.Header {
	if w.header == nil {
		w.header = make(http.Header)
	}
	return w.header
}

func (*errResponseWriter) Write([]byte) (int, error) {
	return 0, errors.New("write failed")
}

func (*errResponseWriter) WriteHeader(int) {}

// parseSSEEvents extracts the JSON payloads from the `data:` frames in the
// supplied SSE body. Each returned map corresponds to one AG-UI event.
// Non-`data:` framing lines (`id:`, blank separators) are ignored, matching
// how an SSE client iterates frames.
//
// Per the SSE spec the colon may be followed by one optional space; both
// `data: {...}` and `data:{...}` are valid frames a server may emit. We
// accept either so the tests do not regress if the AG-UI SDK ever changes
// its framing whitespace.
func parseSSEEvents(t *testing.T, body string) []map[string]any {
	t.Helper()
	var events []map[string]any
	for _, line := range strings.Split(body, "\n") {
		data, ok := strings.CutPrefix(line, "data:")
		if !ok {
			continue
		}
		// The SSE spec allows exactly one optional space after `data:`
		// — strip it if present so the JSON parser sees the payload
		// regardless of which framing variant the writer chose.
		data = strings.TrimPrefix(data, " ")
		var event map[string]any
		require.NoError(t, json.Unmarshal([]byte(data), &event), "could not parse SSE frame %q", data)
		events = append(events, event)
	}
	return events
}

func eventTypes(events []map[string]any) []string {
	types := make([]string, 0, len(events))
	for _, e := range events {
		if t, ok := e["type"].(string); ok {
			types = append(types, t)
		}
	}
	return types
}

func TestParseSSEEventsAcceptsDataPrefixWithOrWithoutSpace(t *testing.T) {
	// Per the SSE spec, the colon may be followed by exactly one optional
	// space. AG-UI's SDK writer currently emits "data: {...}\n\n" but the
	// parser must tolerate the no-space variant so the tests do not
	// regress if the SDK ever switches framing whitespace.
	body := "data: {\"type\":\"RUN_STARTED\",\"runId\":\"r1\"}\n\n" +
		"data:{\"type\":\"RUN_FINISHED\",\"runId\":\"r2\"}\n\n"

	events := parseSSEEvents(t, body)
	require.Len(t, events, 2,
		"both `data:` and `data: ` framing variants must be recognised by the parser")
	assert.Equal(t, "RUN_STARTED", events[0]["type"])
	assert.Equal(t, "RUN_FINISHED", events[1]["type"])
}

func TestAGUIRunEmitWritesSSEFrame(t *testing.T) {
	rec := httptest.NewRecorder()
	c := newAGUIRun(context.Background(), rec, "thread-1", "run-1")

	c.emit(aguievents.NewRunStartedEvent("thread-1", "run-1"))

	body := rec.Body.String()
	assert.Contains(t, body, "data: ", "emit should produce a data: SSE line")
	events := parseSSEEvents(t, body)
	require.Len(t, events, 1)
	assert.Equal(t, "RUN_STARTED", events[0]["type"])
}

func TestAGUIRunEmitContextDoneSetsClosedFlag(t *testing.T) {
	rec := httptest.NewRecorder()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	c := newAGUIRun(ctx, rec, "thread-1", "run-1")

	c.emit(aguievents.NewRunStartedEvent("thread-1", "run-1"))

	assert.True(t, c.closed, "expected client to be closed when context is cancelled")
	assert.Empty(t, rec.Body.String(), "expected no SSE frames after context cancellation")
}

func TestAGUIRunEmitErrorSetsClosedFlag(t *testing.T) {
	c := &aguiRun{
		requestCtx: context.Background(),
		w:          &errResponseWriter{},
		sse:        aguisse.NewSSEWriter(),
	}

	c.emit(aguievents.NewRunStartedEvent("thread-1", "run-1"))

	assert.True(t, c.closed, "expected client to be closed on write error")
}

func TestAGUIRunEmitNoopWhenClosed(t *testing.T) {
	rec := httptest.NewRecorder()
	c := &aguiRun{
		requestCtx: context.Background(),
		w:          rec,
		sse:        aguisse.NewSSEWriter(),
		closed:     true,
	}

	c.emit(aguievents.NewRunStartedEvent("thread-1", "run-1"))

	assert.Empty(t, rec.Body.String(), "expected no writes when already closed")
}

func TestAGUIRunStartRunEmitsRunStarted(t *testing.T) {
	rec := httptest.NewRecorder()
	c := newAGUIRun(context.Background(), rec, "thread-1", "run-1")

	c.startRun()

	events := parseSSEEvents(t, rec.Body.String())
	require.Len(t, events, 1, "expected one RUN_STARTED event")
	assert.Equal(t, "RUN_STARTED", events[0]["type"])
	assert.Equal(t, "thread-1", events[0]["threadId"])
	assert.Equal(t, "run-1", events[0]["runId"])
	assert.Equal(t, "run-1", c.runID, "preserved runID should stay")
	assert.NotEmpty(t, c.messageID, "startRun should allocate a messageID")
}

func TestAGUIRunStartRunGeneratesIdsWhenEmpty(t *testing.T) {
	rec := httptest.NewRecorder()
	c := newAGUIRun(context.Background(), rec, "", "")

	c.startRun()

	assert.NotEmpty(t, c.threadID, "startRun should allocate a threadID when none was provided")
	assert.NotEmpty(t, c.runID, "startRun should allocate a runID when none was provided")
	assert.NotEmpty(t, c.messageID, "startRun should allocate a messageID")

	// Pin the single-nanosecond stem: when all three IDs are allocated by
	// the same startRun call, they must share the suffix so a debugger
	// can see at a glance they came from the same run. Independent
	// time.Now() calls could produce identical values on coarse-clock
	// systems and unrelated-looking values on fast clocks; sharing the
	// stem covers both ends.
	threadSuffix := strings.TrimPrefix(c.threadID, "thread-")
	runSuffix := strings.TrimPrefix(c.runID, "run-")
	msgSuffix := strings.TrimPrefix(c.messageID, "msg-")
	assert.Equal(t, threadSuffix, runSuffix, "threadID and runID must share the nanosecond stem")
	assert.Equal(t, threadSuffix, msgSuffix, "threadID and messageID must share the nanosecond stem")
}

func TestAGUIRunStartRunMintsUniqueStemsAcrossCalls(t *testing.T) {
	// The process-wide atomic counter is the guarantee against coarse-clock
	// collisions: two aguiRuns constructed and startRun-ed in the
	// same nanosecond must still get distinct stems. Calling startRun
	// twice in rapid succession in a single goroutine is the simplest way
	// to exercise that — the timestamp may or may not increment between
	// calls, but the counter must.
	a := newAGUIRun(context.Background(), httptest.NewRecorder(), "", "")
	b := newAGUIRun(context.Background(), httptest.NewRecorder(), "", "")

	a.startRun()
	b.startRun()

	assert.NotEqual(t, a.threadID, b.threadID,
		"two consecutive startRuns must produce different stems even on coarse-clock systems — the seq counter is the safety net")
	assert.NotEqual(t, a.runID, b.runID, "runIDs must be distinct too")
	assert.NotEqual(t, a.messageID, b.messageID, "messageIDs must be distinct too")
}

func TestAGUIRunStartRunDoesNotOverwriteSuppliedIDs(t *testing.T) {
	// When the caller already supplies threadID/runID, startRun must keep
	// them as-is and only allocate the missing messageID. The caller's
	// IDs come from the AG-UI RunAgentInput and are what the frontend
	// uses to correlate the run — overwriting them would orphan events.
	rec := httptest.NewRecorder()
	c := newAGUIRun(context.Background(), rec, "thread-from-frontend", "run-from-frontend")

	c.startRun()

	assert.Equal(t, "thread-from-frontend", c.threadID, "caller-supplied threadID must survive startRun")
	assert.Equal(t, "run-from-frontend", c.runID, "caller-supplied runID must survive startRun")
	assert.NotEmpty(t, c.messageID, "missing messageID must still be allocated")
}

func TestAGUIRunFinishRunClosesOpenTextAndEmitsRunFinished(t *testing.T) {
	rec := httptest.NewRecorder()
	c := newAGUIRun(context.Background(), rec, "thread-1", "run-1")
	c.startRun()
	c.ensureTextStart()

	// AG-UI's RUN_FINISHED schema has no top-level stopReason field, so
	// the sidecar's stop reason is forwarded via the schema-supported
	// `result` payload as `{"stopReason": "<reason>"}`.
	c.finishRun("end_turn")

	events := parseSSEEvents(t, rec.Body.String())
	types := eventTypes(events)
	assert.Equal(t, []string{"RUN_STARTED", "TEXT_MESSAGE_START", "TEXT_MESSAGE_END", "RUN_FINISHED"}, types)
	finishedEvent := events[3]
	_, hasStopReasonAtRoot := finishedEvent["stopReason"]
	assert.False(t, hasStopReasonAtRoot,
		"stopReason must not be a top-level field — the AG-UI schema forbids extras at that level")
	result, ok := finishedEvent["result"].(map[string]any)
	require.True(t, ok, "stopReason must ride inside the schema-supported result payload")
	assert.Equal(t, "end_turn", result["stopReason"])
	assert.False(t, c.textOpen, "textOpen should be reset after finishRun")
	assert.Empty(t, c.messageID, "messageID should be cleared with the text message")
}

func TestAGUIRunFinishRunOmitsResultWhenNoStopReason(t *testing.T) {
	rec := httptest.NewRecorder()
	c := newAGUIRun(context.Background(), rec, "thread-1", "run-1")
	c.startRun()

	c.finishRun("")

	events := parseSSEEvents(t, rec.Body.String())
	require.Len(t, events, 2)
	finishedEvent := events[1]
	assert.Equal(t, "RUN_FINISHED", finishedEvent["type"])
	_, hasResult := finishedEvent["result"]
	assert.False(t, hasResult,
		"result must be omitted when no stop reason is supplied so the event stays compact")
}

func TestAGUIRunFailRunEmitsRunError(t *testing.T) {
	rec := httptest.NewRecorder()
	c := newAGUIRun(context.Background(), rec, "thread-1", "run-1")
	c.startRun()
	c.ensureTextStart()

	c.failRun("boom")

	events := parseSSEEvents(t, rec.Body.String())
	types := eventTypes(events)
	assert.Equal(t, []string{"RUN_STARTED", "TEXT_MESSAGE_START", "TEXT_MESSAGE_END", "RUN_ERROR"}, types)
	assert.Equal(t, "boom", events[3]["message"])
	assert.Equal(t, "run-1", events[3]["runId"], "RUN_ERROR should propagate runId so the frontend can correlate")
	assert.False(t, c.textOpen)
	assert.Empty(t, c.messageID)
}

func TestAGUIRunEnsureTextStartIsIdempotent(t *testing.T) {
	rec := httptest.NewRecorder()
	c := newAGUIRun(context.Background(), rec, "thread-1", "run-1")
	// startRun would normally allocate this; set it directly so the test
	// targets ensureTextStart only. Without a messageID the SDK rejects the
	// event at encode time and emits nothing.
	c.messageID = "msg-1"

	c.ensureTextStart()
	c.ensureTextStart()
	c.ensureTextStart()

	types := eventTypes(parseSSEEvents(t, rec.Body.String()))
	assert.Equal(t, []string{"TEXT_MESSAGE_START"}, types, "ensureTextStart should only emit once per open message")
}

func messageIDs(events []map[string]any, eventType string) []string {
	var ids []string
	for _, e := range events {
		if e["type"] == eventType {
			ids = append(ids, e["messageId"].(string))
		}
	}
	return ids
}

func contentDeltas(events []map[string]any) []string {
	var deltas []string
	for _, e := range events {
		if e["type"] == "TEXT_MESSAGE_CONTENT" {
			deltas = append(deltas, e["delta"].(string))
		}
	}
	return deltas
}

func sendAgentText(t *testing.T, c *aguiRun, text string) {
	t.Helper()
	sendAgentContent(t, c, acp.TextBlock(text))
}

func sendAgentContent(t *testing.T, c *aguiRun, content acp.ContentBlock) {
	t.Helper()
	err := newACPClient(c).SessionUpdate(context.Background(), acp.SessionNotification{
		Update: acp.SessionUpdate{
			AgentMessageChunk: &acp.SessionUpdateAgentMessageChunk{Content: content},
		},
	})
	require.NoError(t, err)
}

func TestAGUIRunWhitespaceAroundToolCall(t *testing.T) {
	toolCall := func(c *aguiRun) {
		t.Helper()
		err := newACPClient(c).SessionUpdate(context.Background(), acp.SessionNotification{
			Update: acp.SessionUpdate{
				ToolCall: &acp.SessionUpdateToolCall{
					ToolCallId: "tool-1",
					Title:      "search_traces",
					Kind:       acp.ToolKindSearch,
				},
			},
		})
		require.NoError(t, err)
	}

	t.Run("whitespace between tool calls before any text is dropped", func(t *testing.T) {
		rec := httptest.NewRecorder()
		c := newAGUIRun(context.Background(), rec, "thread-1", "run-1")
		c.startRun()

		sendAgentText(t, c, "\n")
		toolCall(c)
		sendAgentText(t, c, "\n")
		toolCall(c)
		sendAgentText(t, c, "after")

		events := parseSSEEvents(t, rec.Body.String())
		assert.Equal(t, []string{
			"RUN_STARTED",
			"TOOL_CALL_START",
			"TOOL_CALL_START",
			"TEXT_MESSAGE_START",
			"TEXT_MESSAGE_CONTENT",
		}, eventTypes(events))
		assert.Equal(t, []string{"after"}, contentDeltas(events))
	})

	t.Run("whitespace inside an open message is kept across a tool call", func(t *testing.T) {
		rec := httptest.NewRecorder()
		c := newAGUIRun(context.Background(), rec, "thread-1", "run-1")
		c.startRun()

		sendAgentText(t, c, "Hello")
		sendAgentText(t, c, "\n")
		toolCall(c)
		sendAgentText(t, c, "next")

		assert.Equal(t, []string{"Hello", "\n", "next"}, contentDeltas(parseSSEEvents(t, rec.Body.String())))
	})
}

func TestAGUIRunWhitespaceRunLifecycle(t *testing.T) {
	t.Run("visible text with streamed whitespace", func(t *testing.T) {
		rec := httptest.NewRecorder()
		c := newAGUIRun(context.Background(), rec, "thread-1", "run-1")
		c.startRun()

		for _, chunk := range []string{"\n", "of", " ", "2.23s", "\n\n", "Done"} {
			sendAgentText(t, c, chunk)
		}
		c.finishRun("")

		events := parseSSEEvents(t, rec.Body.String())
		assert.Equal(t, []string{
			"RUN_STARTED",
			"TEXT_MESSAGE_START",
			"TEXT_MESSAGE_CONTENT",
			"TEXT_MESSAGE_CONTENT",
			"TEXT_MESSAGE_CONTENT",
			"TEXT_MESSAGE_CONTENT",
			"TEXT_MESSAGE_CONTENT",
			"TEXT_MESSAGE_END",
			"RUN_FINISHED",
		}, eventTypes(events))
		assert.Equal(t, "of 2.23s\n\nDone", strings.Join(contentDeltas(events), ""))
		assert.False(t, c.textOpen)
	})

	t.Run("only whitespace then finish emits no message", func(t *testing.T) {
		rec := httptest.NewRecorder()
		c := newAGUIRun(context.Background(), rec, "thread-1", "run-1")
		c.startRun()

		sendAgentText(t, c, "\n\n")
		c.finishRun("end_turn")

		events := parseSSEEvents(t, rec.Body.String())
		assert.Equal(t, []string{"RUN_STARTED", "RUN_FINISHED"}, eventTypes(events))
		assert.False(t, c.textOpen)
	})
}

func TestAGUIRunToolCallClosesTextAndStartsNewMessage(t *testing.T) {
	tests := []struct {
		name     string
		toolCall func(t *testing.T, c *aguiRun)
		types    []string
	}{
		{
			name: "ACP tool call",
			toolCall: func(t *testing.T, c *aguiRun) {
				t.Helper()
				err := newACPClient(c).SessionUpdate(context.Background(), acp.SessionNotification{
					Update: acp.SessionUpdate{
						ToolCall: &acp.SessionUpdateToolCall{
							ToolCallId: "tool-1",
							Title:      "search_traces",
							Kind:       acp.ToolKindSearch,
						},
					},
				})
				require.NoError(t, err)
			},
			types: []string{
				"RUN_STARTED",
				"TEXT_MESSAGE_START", "TEXT_MESSAGE_CONTENT", "TEXT_MESSAGE_END",
				"TOOL_CALL_START",
				"TEXT_MESSAGE_START", "TEXT_MESSAGE_CONTENT", "TEXT_MESSAGE_END",
				"RUN_FINISHED",
			},
		},
		{
			name: "UI tool call",
			toolCall: func(_ *testing.T, c *aguiRun) {
				c.EmitUIToolCall("tool-1", "navigate", nil)
			},
			types: []string{
				"RUN_STARTED",
				"TEXT_MESSAGE_START", "TEXT_MESSAGE_CONTENT", "TEXT_MESSAGE_END",
				"TOOL_CALL_START", "TOOL_CALL_END",
				"TEXT_MESSAGE_START", "TEXT_MESSAGE_CONTENT", "TEXT_MESSAGE_END",
				"RUN_FINISHED",
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rec := httptest.NewRecorder()
			c := newAGUIRun(context.Background(), rec, "thread-1", "run-1")
			c.startRun()
			sendAgentText(t, c, "before")
			tt.toolCall(t, c)
			sendAgentText(t, c, "after")
			c.finishRun("")

			events := parseSSEEvents(t, rec.Body.String())
			require.Equal(t, tt.types, eventTypes(events))
			assert.Equal(t, []string{"before", "after"}, contentDeltas(events))

			starts := messageIDs(events, "TEXT_MESSAGE_START")
			require.Len(t, starts, 2)
			assert.NotEqual(t, starts[0], starts[1],
				"the UI keeps one text part per messageId, so reusing the id would append the text to the part before the tool call")
			assert.Equal(t, starts, messageIDs(events, "TEXT_MESSAGE_CONTENT"))
			assert.Equal(t, starts, messageIDs(events, "TEXT_MESSAGE_END"))
		})
	}
}

func TestAGUIRunToolCallWithoutOpenTextEmitsNoTextEnd(t *testing.T) {
	rec := httptest.NewRecorder()
	c := newAGUIRun(context.Background(), rec, "thread-1", "run-1")
	c.startRun()
	c.EmitUIToolCall("tool-1", "navigate", nil)

	assert.Equal(t, []string{"RUN_STARTED", "TOOL_CALL_START", "TOOL_CALL_END"},
		eventTypes(parseSSEEvents(t, rec.Body.String())))
}

func TestAGUIRunUtilityMethods(t *testing.T) {
	assert.Equal(t, "unknown", valueOrUnknown(nil))
	status := acp.ToolCallStatusInProgress
	assert.Equal(t, "in_progress", valueOrUnknown(&status))
}
