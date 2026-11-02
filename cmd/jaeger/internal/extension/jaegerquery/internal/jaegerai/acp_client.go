// Copyright (c) 2026 The Jaeger Authors.
// SPDX-License-Identifier: Apache-2.0

package jaegerai

import (
	"context"
	"errors"
	"time"

	"github.com/coder/acp-go-sdk"
	"go.uber.org/zap"
)

var _ acp.Client = (*acpClient)(nil)

// acpClient is the gateway's side of the ACP connection to the sidecar agent
// for one chat turn. The ACP SDK calls it with the agent's session updates and
// permission requests; it translates each update into events on the turn's
// aguiRun and declines the file and terminal operations the gateway does not
// offer.
type acpClient struct {
	run *aguiRun
}

func newACPClient(run *aguiRun) *acpClient {
	return &acpClient{run: run}
}

// RequestPermission auto-accepts tool calls by selecting the first
// allow-flavoured option the agent offers. The gateway curates which tools
// an agent can invoke through two pre-approved channels:
//   - MCP tools the operator wires up at deploy time (read-only telemetry
//     queries against Jaeger).
//   - UI tools the frontend declares for the chat turn (browser-
//     side AG-UI tools the user is implicitly authorising by sending the
//     prompt).
//
// Filesystem and terminal capabilities are disabled in Initialize, so a
// well-behaved agent will not have other tools to ask about; if one does
// ask, we still accept because the tool itself was already curated upstream.
// AG-UI has no permission-request event shape to bubble a per-call prompt
// up to the browser, so deferring would be useless.
//
// The previous "always cancel" policy was incompatible with agents like the
// Claude Agent SDK, which routes every tool call through canUseTool / a
// session/request_permission RPC; the cancelled outcome aborted every call
// as "Tool use aborted". The Gemini sidecar never hit this because Gemini's
// tool path does not issue ACP permission requests.
//
// If the agent presents no allow-kind option (e.g. only reject choices),
// we fall through to cancelled so a misconfigured agent cannot be coerced
// into a forbidden action.
func (*acpClient) RequestPermission(_ context.Context, req acp.RequestPermissionRequest) (acp.RequestPermissionResponse, error) {
	for _, opt := range req.Options {
		if opt.Kind == acp.PermissionOptionKindAllowOnce || opt.Kind == acp.PermissionOptionKindAllowAlways {
			return acp.RequestPermissionResponse{
				Outcome: acp.RequestPermissionOutcome{
					Selected: &acp.RequestPermissionOutcomeSelected{OptionId: opt.OptionId},
				},
			}, nil
		}
	}
	return acp.RequestPermissionResponse{
		Outcome: acp.RequestPermissionOutcome{
			Cancelled: &acp.RequestPermissionOutcomeCancelled{},
		},
	}, nil
}

// SessionUpdate translates one ACP session update into AG-UI events on the run.
func (c *acpClient) SessionUpdate(_ context.Context, n acp.SessionNotification) error {
	u := n.Update
	if u.AgentMessageChunk != nil && u.AgentMessageChunk.Content.Text != nil {
		c.run.appendText(u.AgentMessageChunk.Content.Text.Text)
	}
	if u.ToolCall != nil {
		// The agent populates ACP Title with the tool name it was offered, which
		// for a UI tool carries UIToolPrefix. Strip it so the frontend sees the
		// name it registered.
		c.run.startToolCall(
			string(u.ToolCall.ToolCallId),
			stripUIToolPrefix(u.ToolCall.Title),
			u.ToolCall.RawInput,
		)
	}
	if u.ToolCallUpdate != nil {
		status := valueOrUnknown(u.ToolCallUpdate.Status)
		finished := status == string(acp.ToolCallStatusCompleted) || status == string(acp.ToolCallStatusFailed)
		c.run.updateToolCall(
			string(u.ToolCallUpdate.ToolCallId),
			u.ToolCallUpdate.RawInput,
			u.ToolCallUpdate.RawOutput,
			finished,
		)
	}
	return nil
}

// The methods below are the acp.Client operations the gateway does not offer:
// filesystem and terminal. They return errors because the chat endpoint
// advertises these capabilities as disabled during Initialize.

var errNotSupported = errors.New("operation not supported by jaeger-ai-gateway")

func (*acpClient) WriteTextFile(context.Context, acp.WriteTextFileRequest) (acp.WriteTextFileResponse, error) {
	return acp.WriteTextFileResponse{}, errNotSupported
}

func (*acpClient) ReadTextFile(context.Context, acp.ReadTextFileRequest) (acp.ReadTextFileResponse, error) {
	return acp.ReadTextFileResponse{}, errNotSupported
}

func (*acpClient) CreateTerminal(context.Context, acp.CreateTerminalRequest) (acp.CreateTerminalResponse, error) {
	return acp.CreateTerminalResponse{}, errNotSupported
}

func (*acpClient) KillTerminal(context.Context, acp.KillTerminalRequest) (acp.KillTerminalResponse, error) {
	return acp.KillTerminalResponse{}, errNotSupported
}

func (*acpClient) ReleaseTerminal(context.Context, acp.ReleaseTerminalRequest) (acp.ReleaseTerminalResponse, error) {
	return acp.ReleaseTerminalResponse{}, errNotSupported
}

func (*acpClient) TerminalOutput(context.Context, acp.TerminalOutputRequest) (acp.TerminalOutputResponse, error) {
	return acp.TerminalOutputResponse{}, errNotSupported
}

func (*acpClient) WaitForTerminalExit(context.Context, acp.WaitForTerminalExitRequest) (acp.WaitForTerminalExitResponse, error) {
	return acp.WaitForTerminalExitResponse{}, errNotSupported
}

// closeACPSession is a best-effort cleanup hook: it tells the agent to
// release the session before the gateway tears down the WebSocket. Meant
// to be invoked via `defer` in ServeHTTP, after defer adapter.Close so it
// fires first (LIFO) while the connection is still open.
//
// The capability gate ensures we never call session/close against agents
// that don't advertise support — they would respond with MethodNotFound.
// WithoutCancel detaches the request context's cancellation so cleanup
// still runs after the client disconnects mid-stream, while preserving
// values such as tracing. The 5s deadline keeps a stuck agent from
// hanging the goroutine indefinitely. Errors are Debug-logged because the
// HTTP response has already been streamed by the time this runs.
func closeACPSession(ctx context.Context, conn *acp.ClientSideConnection, caps acp.AgentCapabilities, sessionID acp.SessionId, logger *zap.Logger) {
	if caps.SessionCapabilities.Close == nil {
		return
	}
	closeCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), 5*time.Second)
	defer cancel()
	_, err := conn.CloseSession(closeCtx, acp.CloseSessionRequest{
		SessionId: sessionID,
	})
	if err != nil {
		logger.Debug("session/close failed", zap.String("session_id", string(sessionID)), zap.Error(err))
	}
}
