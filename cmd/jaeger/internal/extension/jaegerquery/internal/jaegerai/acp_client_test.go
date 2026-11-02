// Copyright (c) 2026 The Jaeger Authors.
// SPDX-License-Identifier: Apache-2.0

package jaegerai

import (
	"context"
	"net/http/httptest"
	"testing"

	"github.com/coder/acp-go-sdk"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestACPClientRequestPermissionAcceptsAllowOnce(t *testing.T) {
	// Agents like the Claude Agent SDK route every tool call through a
	// session/request_permission RPC. The gateway already curates which
	// tools the agent can invoke (operator-configured MCP + frontend-
	// declared contextual tools), and AG-UI has no per-call permission
	// event to bubble up — so the gateway must select an allow option
	// rather than cancel, which Claude treats as "Tool use aborted".
	c := &acpClient{}

	resp, err := c.RequestPermission(context.Background(), acp.RequestPermissionRequest{
		Options: []acp.PermissionOption{{
			OptionId: "opt-1",
			Name:     "allow once",
			Kind:     acp.PermissionOptionKindAllowOnce,
		}},
	})
	require.NoError(t, err)
	require.NotNil(t, resp.Outcome.Selected, "expected allow-kind option to be selected")
	require.Nil(t, resp.Outcome.Cancelled, "must not cancel when an allow option is offered")
	assert.Equal(t, acp.PermissionOptionId("opt-1"), resp.Outcome.Selected.OptionId)
}

func TestACPClientRequestPermissionPicksFirstAllowOption(t *testing.T) {
	// Agents typically present a list like [allow_once, allow_always,
	// reject_once]. Order-preserving "first allow wins" keeps the agent's
	// suggested ordering authoritative — the gateway has no preference
	// between scopes, so don't impose one.
	c := &acpClient{}

	resp, err := c.RequestPermission(context.Background(), acp.RequestPermissionRequest{
		Options: []acp.PermissionOption{
			{OptionId: "reject", Name: "reject", Kind: acp.PermissionOptionKindRejectOnce},
			{OptionId: "once", Name: "allow once", Kind: acp.PermissionOptionKindAllowOnce},
			{OptionId: "always", Name: "allow always", Kind: acp.PermissionOptionKindAllowAlways},
		},
	})
	require.NoError(t, err)
	require.NotNil(t, resp.Outcome.Selected)
	assert.Equal(t, acp.PermissionOptionId("once"), resp.Outcome.Selected.OptionId,
		"first allow-kind option must be chosen, skipping any preceding reject options")
}

func TestACPClientRequestPermissionAcceptsAllowAlwaysWhenNoOnce(t *testing.T) {
	// Some agents may only offer allow_always (e.g. for tools they consider
	// safe to remember). Accept that too — both kinds are "yes."
	c := &acpClient{}

	resp, err := c.RequestPermission(context.Background(), acp.RequestPermissionRequest{
		Options: []acp.PermissionOption{{
			OptionId: "always",
			Name:     "allow always",
			Kind:     acp.PermissionOptionKindAllowAlways,
		}},
	})
	require.NoError(t, err)
	require.NotNil(t, resp.Outcome.Selected)
	assert.Equal(t, acp.PermissionOptionId("always"), resp.Outcome.Selected.OptionId)
}

func TestACPClientRequestPermissionCancelsWhenNoOptions(t *testing.T) {
	// Defensive: with no options to choose from there's nothing to accept,
	// so fall through to cancelled rather than fabricate an option id.
	c := &acpClient{}

	resp, err := c.RequestPermission(context.Background(), acp.RequestPermissionRequest{})
	require.NoError(t, err)
	require.NotNil(t, resp.Outcome.Cancelled, "no options → cancelled")
	require.Nil(t, resp.Outcome.Selected)
}

func TestACPClientRequestPermissionCancelsWhenOnlyRejectOptions(t *testing.T) {
	// A misconfigured or hostile agent that only presents reject options
	// cannot be coerced into an allow — the gateway falls through to
	// cancelled so the tool call is aborted instead of forced.
	c := &acpClient{}

	resp, err := c.RequestPermission(context.Background(), acp.RequestPermissionRequest{
		Options: []acp.PermissionOption{
			{OptionId: "reject-once", Kind: acp.PermissionOptionKindRejectOnce},
			{OptionId: "reject-always", Kind: acp.PermissionOptionKindRejectAlways},
		},
	})
	require.NoError(t, err)
	require.NotNil(t, resp.Outcome.Cancelled)
	require.Nil(t, resp.Outcome.Selected, "must not promote a reject option to an allow")
}

func TestACPClientUnsupportedOperationsReturnError(t *testing.T) {
	c := &acpClient{}

	_, err := c.WriteTextFile(context.Background(), acp.WriteTextFileRequest{})
	require.ErrorIs(t, err, errNotSupported)

	_, err = c.ReadTextFile(context.Background(), acp.ReadTextFileRequest{Path: "/tmp/nope"})
	require.ErrorIs(t, err, errNotSupported)

	_, err = c.CreateTerminal(context.Background(), acp.CreateTerminalRequest{})
	require.ErrorIs(t, err, errNotSupported)

	_, err = c.KillTerminal(context.Background(), acp.KillTerminalRequest{})
	require.ErrorIs(t, err, errNotSupported)

	_, err = c.ReleaseTerminal(context.Background(), acp.ReleaseTerminalRequest{})
	require.ErrorIs(t, err, errNotSupported)

	_, err = c.TerminalOutput(context.Background(), acp.TerminalOutputRequest{})
	require.ErrorIs(t, err, errNotSupported)

	_, err = c.WaitForTerminalExit(context.Background(), acp.WaitForTerminalExitRequest{})
	require.ErrorIs(t, err, errNotSupported)
}

func TestACPClientSessionUpdateAgentMessageChunkEmitsTextContent(t *testing.T) {
	rec := httptest.NewRecorder()
	c := newAGUIRun(context.Background(), rec, "thread-1", "run-1")
	c.startRun()

	err := newACPClient(c).SessionUpdate(context.Background(), acp.SessionNotification{
		Update: acp.SessionUpdate{
			AgentMessageChunk: &acp.SessionUpdateAgentMessageChunk{Content: acp.TextBlock("hello world")},
		},
	})
	require.NoError(t, err)

	events := parseSSEEvents(t, rec.Body.String())
	types := eventTypes(events)
	assert.Equal(t, []string{"RUN_STARTED", "TEXT_MESSAGE_START", "TEXT_MESSAGE_CONTENT"}, types)

	contentEvent := events[2]
	assert.Equal(t, "hello world", contentEvent["delta"])
	assert.Equal(t, c.messageID, contentEvent["messageId"])
}

func TestACPClientSessionUpdateAgentMessageChunkSkipsEmptyText(t *testing.T) {
	// AG-UI's TextMessageContentEvent forbids empty deltas — its SDK encoder
	// rejects them at write time, which would mark the SSE stream closed and
	// strand the browser bubble after TEXT_MESSAGE_START. ACP itself permits
	// empty-text chunks (Claude maps content_block_start lifecycle events
	// that way), so the gateway must filter them at the boundary. No
	// TEXT_MESSAGE_START fires either: pairing START with content that
	// never arrives breaks the AG-UI invariant START → CONTENT+ → END.
	rec := httptest.NewRecorder()
	c := newAGUIRun(context.Background(), rec, "thread-1", "run-1")

	err := newACPClient(c).SessionUpdate(context.Background(), acp.SessionNotification{
		Update: acp.SessionUpdate{
			AgentMessageChunk: &acp.SessionUpdateAgentMessageChunk{Content: acp.TextBlock("")},
		},
	})
	require.NoError(t, err)
	require.Empty(t, rec.Body.String(),
		"empty-text agent_message_chunk must not produce any SSE events")
	assert.False(t, c.textOpen,
		"TEXT_MESSAGE_START must not be emitted when no content will follow")
}

func TestACPClientSessionUpdateKeepsWhitespaceChunks(t *testing.T) {
	// ACP agents such as Claude Code (via claude-agent-acp) stream spaces
	// and newlines as their own chunks. Dropping those glues tokens
	// ("of" + "2.23s" → "of2.23s") and erases paragraph breaks. Empty
	// chunks are still skipped, and a whitespace-only chunk that arrives
	// before any visible text is dropped so it does not open a message.
	tests := []struct {
		name   string
		chunks []string
		deltas []string
	}{
		{
			name:   "space between tokens",
			chunks: []string{"of", " ", "2.23s"},
			deltas: []string{"of", " ", "2.23s"},
		},
		{
			name:   "paragraph break",
			chunks: []string{"para one.", "\n\n", "Para two"},
			deltas: []string{"para one.", "\n\n", "Para two"},
		},
		{
			name:   "empty chunks skipped between real deltas",
			chunks: []string{"", "of", "", " ", "", "2.23s", ""},
			deltas: []string{"of", " ", "2.23s"},
		},
		{
			name:   "only empty chunks emit no text",
			chunks: []string{"", ""},
		},
		{
			name:   "leading whitespace does not open a message",
			chunks: []string{" ", "  \t  ", "\n", "\r\n"},
		},
		{
			name:   "leading whitespace is dropped before the first visible chunk",
			chunks: []string{"\n\n", "Para two"},
			deltas: []string{"Para two"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rec := httptest.NewRecorder()
			c := newAGUIRun(context.Background(), rec, "thread-1", "run-1")
			c.startRun()

			for _, chunk := range tt.chunks {
				sendAgentText(t, c, chunk)
			}

			events := parseSSEEvents(t, rec.Body.String())
			assert.Equal(t, tt.deltas, contentDeltas(events))

			if tt.deltas == nil {
				assert.False(t, c.textOpen,
					"TEXT_MESSAGE_START must not open when no visible text arrived")
				assert.Equal(t, []string{"RUN_STARTED"}, eventTypes(events))
				return
			}
			assert.True(t, c.textOpen)
			require.GreaterOrEqual(t, len(events), 2)
			assert.Equal(t, "TEXT_MESSAGE_START", events[1]["type"])
		})
	}
}

func TestACPClientSessionUpdateNonTextChunkDoesNotPanic(t *testing.T) {
	rec := httptest.NewRecorder()
	c := newAGUIRun(context.Background(), rec, "thread-1", "run-1")
	c.startRun()

	require.NotPanics(t, func() {
		sendAgentContent(t, c, acp.ContentBlock{})
		sendAgentContent(t, c, acp.ImageBlock("aGVsbG8=", "image/png"))
	})

	events := parseSSEEvents(t, rec.Body.String())
	assert.Equal(t, []string{"RUN_STARTED"}, eventTypes(events))
	assert.False(t, c.textOpen)
}

func TestACPClientSessionUpdateAgentMessageChunkPreservesInternalWhitespace(t *testing.T) {
	// Whitespace inside a chunk is meaningful glue between streamed
	// deltas — e.g. when an agent emits "Hello" then " world" separately,
	// the leading space must survive so the assembled message reads
	// "Hello world". The filter drops only empty chunks; it does not
	// trim a delta it forwards.
	rec := httptest.NewRecorder()
	c := newAGUIRun(context.Background(), rec, "thread-1", "run-1")
	c.startRun()

	err := newACPClient(c).SessionUpdate(context.Background(), acp.SessionNotification{
		Update: acp.SessionUpdate{
			AgentMessageChunk: &acp.SessionUpdateAgentMessageChunk{Content: acp.TextBlock(" world ")},
		},
	})
	require.NoError(t, err)

	events := parseSSEEvents(t, rec.Body.String())
	require.Equal(t,
		[]string{"RUN_STARTED", "TEXT_MESSAGE_START", "TEXT_MESSAGE_CONTENT"},
		eventTypes(events))
	assert.Equal(t, " world ", events[2]["delta"],
		"non-blank chunks must be forwarded verbatim, preserving leading/trailing whitespace")
}

func TestACPClientSessionUpdateToolCallEmitsStartAndArgs(t *testing.T) {
	rec := httptest.NewRecorder()
	c := newAGUIRun(context.Background(), rec, "thread-1", "run-1")

	err := newACPClient(c).SessionUpdate(context.Background(), acp.SessionNotification{
		Update: acp.SessionUpdate{
			ToolCall: &acp.SessionUpdateToolCall{
				ToolCallId: "tool-1",
				Title:      "search_traces",
				Kind:       acp.ToolKindSearch,
				RawInput:   map[string]any{"service": "checkout"},
			},
		},
	})
	require.NoError(t, err)

	events := parseSSEEvents(t, rec.Body.String())
	require.Len(t, events, 2)

	startEvent := events[0]
	assert.Equal(t, "TOOL_CALL_START", startEvent["type"])
	assert.Equal(t, "tool-1", startEvent["toolCallId"])
	assert.Equal(t, "search_traces", startEvent["toolCallName"],
		"AG-UI TOOL_CALL_START requires toolCallName")

	argsEvent := events[1]
	assert.Equal(t, "TOOL_CALL_ARGS", argsEvent["type"])
	assert.Equal(t, "tool-1", argsEvent["toolCallId"])
	// AG-UI streams args as text deltas; the whole input is emitted as one
	// JSON-encoded delta because the sidecar delivers args atomically.
	delta, ok := argsEvent["delta"].(string)
	require.True(t, ok, "TOOL_CALL_ARGS.delta must be a string")
	assert.JSONEq(t, `{"service":"checkout"}`, delta,
		"TOOL_CALL_ARGS.delta must be the JSON-encoded args string")
}

func TestACPClientSessionUpdateToolCallStripsUIPrefixForName(t *testing.T) {
	rec := httptest.NewRecorder()
	c := newAGUIRun(context.Background(), rec, "thread-1", "run-1")

	// The agent reports progress under the name the MCP endpoint advertised,
	// which for a UI tool carries UIToolPrefix. The frontend registered it under
	// the bare name, so TOOL_CALL_START.toolCallName must be stripped.
	err := newACPClient(c).SessionUpdate(context.Background(), acp.SessionNotification{
		Update: acp.SessionUpdate{
			ToolCall: &acp.SessionUpdateToolCall{
				ToolCallId: "tool-2",
				Title:      UIToolPrefix + "render_chart",
				Kind:       acp.ToolKindOther,
			},
		},
	})
	require.NoError(t, err)

	events := parseSSEEvents(t, rec.Body.String())
	require.Len(t, events, 1)
	assert.Equal(t, "render_chart", events[0]["toolCallName"],
		"contextual tool names must have %q stripped before being sent to the frontend", UIToolPrefix)
}

func TestACPClientSessionUpdateToolCallUpdateEmitsResultAndEnd(t *testing.T) {
	rec := httptest.NewRecorder()
	c := newAGUIRun(context.Background(), rec, "thread-1", "run-1")

	completed := acp.ToolCallStatusCompleted
	err := newACPClient(c).SessionUpdate(context.Background(), acp.SessionNotification{
		Update: acp.SessionUpdate{
			ToolCallUpdate: &acp.SessionToolCallUpdate{
				ToolCallId: "tool-1",
				RawInput:   map[string]any{"refined": true},
				RawOutput:  map[string]any{"hits": 42},
				Status:     &completed,
			},
		},
	})
	require.NoError(t, err)

	events := parseSSEEvents(t, rec.Body.String())
	types := eventTypes(events)
	assert.Equal(t, []string{"TOOL_CALL_ARGS", "TOOL_CALL_RESULT", "TOOL_CALL_END"}, types)

	argsEvent := events[0]
	assert.Equal(t, `{"refined":true}`, argsEvent["delta"])

	resultEvent := events[1]
	assert.Equal(t, "tool-1", resultEvent["toolCallId"])
	assert.Equal(t, "tool-msg-tool-1", resultEvent["messageId"],
		"AG-UI TOOL_CALL_RESULT requires messageId")
	assert.Equal(t, "tool", resultEvent["role"])
	assert.Equal(t, `{"hits":42}`, resultEvent["content"],
		"non-MCP-envelope outputs should be JSON-encoded into the content string")
	_, hasResult := resultEvent["result"]
	assert.False(t, hasResult, "legacy nested result field must not be present")

	assert.Equal(t, "tool-1", events[2]["toolCallId"])
}

func TestACPClientSessionUpdateToolCallUpdateFlattensMCPEnvelope(t *testing.T) {
	rec := httptest.NewRecorder()
	c := newAGUIRun(context.Background(), rec, "thread-1", "run-1")

	completed := acp.ToolCallStatusCompleted
	// Sidecar forwards MCP CallToolResult envelopes verbatim. The handler
	// must concatenate the text blocks so AG-UI receives content as a string.
	err := newACPClient(c).SessionUpdate(context.Background(), acp.SessionNotification{
		Update: acp.SessionUpdate{
			ToolCallUpdate: &acp.SessionToolCallUpdate{
				ToolCallId: "search_traces-1",
				RawOutput: map[string]any{
					"content": []any{
						map[string]any{"type": "text", "text": `{"traces":[]}`},
					},
					"isError":           false,
					"structuredContent": map[string]any{"traces": []any{}},
				},
				Status: &completed,
			},
		},
	})
	require.NoError(t, err)

	events := parseSSEEvents(t, rec.Body.String())
	require.Len(t, events, 2, "expected TOOL_CALL_RESULT and TOOL_CALL_END")

	resultEvent := events[0]
	assert.Equal(t, "TOOL_CALL_RESULT", resultEvent["type"])
	content, ok := resultEvent["content"].(string)
	require.True(t, ok, "TOOL_CALL_RESULT.content must be a string")
	assert.JSONEq(t, `{"traces":[]}`, content,
		"MCP text block should become the top-level content string")
	assert.Equal(t, "tool-msg-search_traces-1", resultEvent["messageId"])
	assert.Equal(t, "tool", resultEvent["role"])
}

func TestACPClientSessionUpdateToolCallUpdateSkipsEndForInProgress(t *testing.T) {
	rec := httptest.NewRecorder()
	c := newAGUIRun(context.Background(), rec, "thread-1", "run-1")

	inProgress := acp.ToolCallStatusInProgress
	err := newACPClient(c).SessionUpdate(context.Background(), acp.SessionNotification{
		Update: acp.SessionUpdate{
			ToolCallUpdate: &acp.SessionToolCallUpdate{
				ToolCallId: "tool-1",
				Status:     &inProgress,
			},
		},
	})
	require.NoError(t, err)

	assert.Empty(t, parseSSEEvents(t, rec.Body.String()),
		"in_progress tool call updates with no content should emit no SSE events")
}
