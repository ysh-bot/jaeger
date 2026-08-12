// Copyright (c) 2026 The Jaeger Authors.
// SPDX-License-Identifier: Apache-2.0

package core

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	esquery "github.com/jaegertracing/jaeger/internal/storage/elasticsearch/query"
	"github.com/jaegertracing/jaeger/internal/storage/v2/api/tracestore"
	"github.com/jaegertracing/jaeger/internal/storage/v2/elasticsearch/tracestore/core/dbmodel"
)

// spanAttr, resourceAttr, eventAttr, and unqualifiedAttr spell the reference kinds the
// tests below compare against, so a case reads as the filter a caller would send.
func spanAttr(name string) *tracestore.Reference {
	return &tracestore.Reference{Name: name, Level: tracestore.LevelSpan, Attr: true}
}

func resourceAttr(name string) *tracestore.Reference {
	return &tracestore.Reference{Name: name, Level: tracestore.LevelResource, Attr: true}
}

func eventAttr(name string) *tracestore.Reference {
	return &tracestore.Reference{Name: name, Level: tracestore.LevelEvent, Attr: true}
}

func unqualifiedAttr(name string) *tracestore.Reference {
	return &tracestore.Reference{Name: name}
}

func scalar(value string) *tracestore.Scalar {
	return &tracestore.Scalar{Value: value}
}

func call(op tracestore.Operator, args ...tracestore.Expression) *tracestore.Call {
	return &tracestore.Call{Op: op, Args: args}
}

// TestFilterCapabilities pins what this reader declares it can serve, because the query
// service refuses a filter on the strength of that declaration alone: a level or operator
// missing here is one no caller can reach, and one listed here that buildFilterQuery cannot
// lower would reach the reader and fail late.
func TestFilterCapabilities(t *testing.T) {
	caps := FilterCapabilities()
	assert.Equal(t, []tracestore.Level{
		tracestore.LevelSpan,
		tracestore.LevelResource,
		tracestore.LevelEvent,
	}, caps.Levels)
	for _, op := range []tracestore.Operator{
		tracestore.OpAnd, tracestore.OpOr, tracestore.OpNot,
		tracestore.OpEq, tracestore.OpNe, tracestore.OpGt, tracestore.OpLt,
		tracestore.OpGte, tracestore.OpLte, tracestore.OpRegex, tracestore.OpExists,
		tracestore.OpIn, tracestore.OpNotIn,
	} {
		assert.True(t, caps.SupportsOperator(op), "expected %q to be declared", op)
	}
	assert.False(t, caps.SupportsOperator(tracestore.OpSome),
		"correlated matching over a span's events is not implemented")
	assert.False(t, caps.SupportsLevel(tracestore.LevelInstrumentation))
	assert.False(t, caps.SupportsLevel(tracestore.LevelLink))
}

func TestBuildFilterQuery(t *testing.T) {
	tests := []struct {
		name   string
		filter *tracestore.Call
		want   string
	}{
		{
			name:   "unqualified attribute searches the span and resource levels",
			filter: call(tracestore.OpEq, unqualifiedAttr("http.status_code"), scalar("500")),
			want: `{"bool":{"should":[
				{"term":{"tag.http@status_code":"500"}},
				{"term":{"process.tag.http@status_code":"500"}},
				{"nested":{"path":"tags","query":{"bool":{"must":[
					{"term":{"tags.key":"http.status_code"}},
					{"term":{"tags.value":"500"}}]}}}},
				{"nested":{"path":"process.tags","query":{"bool":{"must":[
					{"term":{"process.tags.key":"http.status_code"}},
					{"term":{"process.tags.value":"500"}}]}}}}]}}`,
		},
		{
			name:   "span level searches only the span's own attributes",
			filter: call(tracestore.OpEq, spanAttr("component"), scalar("grpc")),
			want: `{"bool":{"should":[
				{"term":{"tag.component":"grpc"}},
				{"nested":{"path":"tags","query":{"bool":{"must":[
					{"term":{"tags.key":"component"}},
					{"term":{"tags.value":"grpc"}}]}}}}]}}`,
		},
		{
			name:   "resource level searches only the process attributes",
			filter: call(tracestore.OpEq, resourceAttr("deployment.environment"), scalar("staging")),
			want: `{"bool":{"should":[
				{"term":{"process.tag.deployment@environment":"staging"}},
				{"nested":{"path":"process.tags","query":{"bool":{"must":[
					{"term":{"process.tags.key":"deployment.environment"}},
					{"term":{"process.tags.value":"staging"}}]}}}}]}}`,
		},
		{
			name:   "event level has one location, so it needs no disjunction",
			filter: call(tracestore.OpEq, eventAttr("exception.type"), scalar("IOError")),
			want: `{"nested":{"path":"logs.fields","query":{"bool":{"must":[
				{"term":{"logs.fields.key":"exception.type"}},
				{"term":{"logs.fields.value":"IOError"}}]}}}}`,
		},
		{
			name:   "event.name reads the logs.fields entry the write path stores it in",
			filter: call(tracestore.OpEq, tracestore.EventName.Ref(), scalar("exception")),
			want: `{"nested":{"path":"logs.fields","query":{"bool":{"must":[
				{"term":{"logs.fields.key":"event"}},
				{"term":{"logs.fields.value":"exception"}}]}}}}`,
		},
		{
			name:   "span.name is the operation name",
			filter: call(tracestore.OpEq, tracestore.SpanName.Ref(), scalar("/api/v3/traces")),
			want:   `{"term":{"operationName":"/api/v3/traces"}}`,
		},
		{
			name:   "resource.service is the service name",
			filter: call(tracestore.OpEq, tracestore.ResourceService.Ref(), scalar("cart")),
			want:   `{"term":{"process.serviceName":"cart"}}`,
		},
		{
			name:   "span.duration compares microseconds against a value carrying its unit",
			filter: call(tracestore.OpGt, tracestore.SpanDuration.Ref(), scalar("2s")),
			want:   `{"range":{"duration":{"gt":2000000}}}`,
		},
		{
			name:   "gte on the duration",
			filter: call(tracestore.OpGte, tracestore.SpanDuration.Ref(), scalar("1500ms")),
			want:   `{"range":{"duration":{"gte":1500000}}}`,
		},
		{
			name:   "lt on the duration",
			filter: call(tracestore.OpLt, tracestore.SpanDuration.Ref(), scalar("1m")),
			want:   `{"range":{"duration":{"lt":60000000}}}`,
		},
		{
			name:   "lte on the duration",
			filter: call(tracestore.OpLte, tracestore.SpanDuration.Ref(), scalar("500us")),
			want:   `{"range":{"duration":{"lte":500}}}`,
		},
		{
			name:   "eq on the duration",
			filter: call(tracestore.OpEq, tracestore.SpanDuration.Ref(), scalar("3s")),
			want:   `{"term":{"duration":3000000}}`,
		},
		{
			name:   "regex compares a pattern rather than a value",
			filter: call(tracestore.OpRegex, tracestore.SpanName.Ref(), scalar("GET .*")),
			want:   `{"regexp":{"operationName":{"value":"GET .*"}}}`,
		},
		{
			name:   "regex on an attribute reaches every location the attribute lives in",
			filter: call(tracestore.OpRegex, spanAttr("http.route"), scalar("/api/.*")),
			want: `{"bool":{"should":[
				{"regexp":{"tag.http@route":{"value":"/api/.*"}}},
				{"nested":{"path":"tags","query":{"bool":{"must":[
					{"term":{"tags.key":"http.route"}},
					{"regexp":{"tags.value":{"value":"/api/.*"}}}]}}}}]}}`,
		},
		{
			name:   "exists tests the key alone",
			filter: call(tracestore.OpExists, eventAttr("exception.stacktrace")),
			want: `{"nested":{"path":"logs.fields",
				"query":{"term":{"logs.fields.key":"exception.stacktrace"}}}}`,
		},
		{
			name:   "exists on an attribute tests the key in every location it may live in",
			filter: call(tracestore.OpExists, spanAttr("http.route")),
			want: `{"bool":{"should":[
				{"exists":{"field":"tag.http@route"}},
				{"nested":{"path":"tags","query":{"term":{"tags.key":"http.route"}}}}]}}`,
		},
		{
			name:   "exists on event.name tests the key the write path stores it under",
			filter: call(tracestore.OpExists, tracestore.EventName.Ref()),
			want: `{"nested":{"path":"logs.fields",
				"query":{"term":{"logs.fields.key":"event"}}}}`,
		},
		{
			name:   "exists on a built-in field tests its own field",
			filter: call(tracestore.OpExists, tracestore.SpanDuration.Ref()),
			want:   `{"exists":{"field":"duration"}}`,
		},
		{
			name:   "exists on the operation name",
			filter: call(tracestore.OpExists, tracestore.SpanName.Ref()),
			want:   `{"exists":{"field":"operationName"}}`,
		},
		{
			name:   "exists on the service name",
			filter: call(tracestore.OpExists, tracestore.ResourceService.Ref()),
			want:   `{"exists":{"field":"process.serviceName"}}`,
		},
		{
			name: "and becomes the must clause",
			filter: call(tracestore.OpAnd,
				call(tracestore.OpEq, tracestore.ResourceService.Ref(), scalar("cart")),
				call(tracestore.OpGt, tracestore.SpanDuration.Ref(), scalar("2s"))),
			want: `{"bool":{"must":[
				{"term":{"process.serviceName":"cart"}},
				{"range":{"duration":{"gt":2000000}}}]}}`,
		},
		{
			// A one-argument conjunction is not a filter a caller can send through the query
			// service, which validates arity first, but it can arrive from a remote-storage
			// client. It means the predicate it wraps, and lowers to it.
			name: "a conjunction of one predicate is that predicate",
			filter: call(tracestore.OpAnd,
				call(tracestore.OpEq, tracestore.ResourceService.Ref(), scalar("cart"))),
			want: `{"term":{"process.serviceName":"cart"}}`,
		},
		{
			name: "or becomes the should clause",
			filter: call(tracestore.OpOr,
				call(tracestore.OpEq, tracestore.ResourceService.Ref(), scalar("cart")),
				call(tracestore.OpEq, tracestore.ResourceService.Ref(), scalar("checkout"))),
			want: `{"bool":{"should":[
				{"term":{"process.serviceName":"cart"}},
				{"term":{"process.serviceName":"checkout"}}]}}`,
		},
		{
			name: "not becomes the must_not clause, which also matches a span missing the reference",
			filter: call(tracestore.OpNot,
				call(tracestore.OpEq, tracestore.ResourceService.Ref(), scalar("healthcheck"))),
			want: `{"bool":{"must_not":{"term":{"process.serviceName":"healthcheck"}}}}`,
		},
		{
			name: "nesting composes to nested bool queries",
			filter: call(tracestore.OpAnd,
				call(tracestore.OpEq, tracestore.ResourceService.Ref(), scalar("cart")),
				call(tracestore.OpOr,
					call(tracestore.OpEq, tracestore.SpanName.Ref(), scalar("a")),
					call(tracestore.OpNot,
						call(tracestore.OpEq, tracestore.SpanName.Ref(), scalar("b"))))),
			want: `{"bool":{"must":[
				{"term":{"process.serviceName":"cart"}},
				{"bool":{"should":[
					{"term":{"operationName":"a"}},
					{"bool":{"must_not":{"term":{"operationName":"b"}}}}]}}]}}`,
		},
		{
			name:   "ne requires the reference to be present, so an absent one does not match",
			filter: call(tracestore.OpNe, eventAttr("exception.type"), scalar("IOError")),
			want: `{"bool":{
				"must":{"nested":{"path":"logs.fields",
					"query":{"term":{"logs.fields.key":"exception.type"}}}},
				"must_not":{"nested":{"path":"logs.fields","query":{"bool":{"must":[
					{"term":{"logs.fields.key":"exception.type"}},
					{"term":{"logs.fields.value":"IOError"}}]}}}}}}`,
		},
		{
			name:   "ne on a built-in field guards on the field's own presence",
			filter: call(tracestore.OpNe, tracestore.ResourceService.Ref(), scalar("cart")),
			want: `{"bool":{
				"must":{"exists":{"field":"process.serviceName"}},
				"must_not":{"term":{"process.serviceName":"cart"}}}}`,
		},
		{
			name: "in is the disjunction of equalities it stands for",
			filter: call(tracestore.OpIn, tracestore.ResourceService.Ref(),
				&tracestore.List{Values: []string{"cart", "checkout"}}),
			want: `{"bool":{"should":[
				{"term":{"process.serviceName":"cart"}},
				{"term":{"process.serviceName":"checkout"}}]}}`,
		},
		{
			name: "a single-member list needs no disjunction",
			filter: call(tracestore.OpIn, tracestore.ResourceService.Ref(),
				&tracestore.List{Values: []string{"cart"}}),
			want: `{"term":{"process.serviceName":"cart"}}`,
		},
		{
			name: "not_in requires the reference to be present, like ne",
			filter: call(tracestore.OpNotIn, tracestore.ResourceService.Ref(),
				&tracestore.List{Values: []string{"cart", "checkout"}}),
			want: `{"bool":{
				"must":{"exists":{"field":"process.serviceName"}},
				"must_not":{"bool":{"should":[
					{"term":{"process.serviceName":"cart"}},
					{"term":{"process.serviceName":"checkout"}}]}}}}`,
		},
		{
			name:   "error=true matches the tag the write path records",
			filter: call(tracestore.OpEq, spanAttr("error"), scalar("true")),
			want: `{"bool":{"should":[
				{"term":{"tag.error":"true"}},
				{"nested":{"path":"tags","query":{"bool":{"must":[
					{"term":{"tags.key":"error"}},
					{"term":{"tags.value":"true"}}]}}}}]}}`,
		},
		{
			name:   "error=false excludes error=true, because a span that succeeded carries no tag",
			filter: call(tracestore.OpEq, spanAttr("error"), scalar("false")),
			want: `{"bool":{"must_not":{"bool":{"should":[
				{"term":{"tag.error":"true"}},
				{"nested":{"path":"tags","query":{"bool":{"must":[
					{"term":{"tags.key":"error"}},
					{"term":{"tags.value":"true"}}]}}}}]}}}}`,
		},
		{
			name:   "an unqualified error=0 is read the same way as error=false",
			filter: call(tracestore.OpEq, unqualifiedAttr("error"), scalar("0")),
			want: `{"bool":{"must_not":{"bool":{"should":[
				{"term":{"tag.error":"true"}},
				{"term":{"process.tag.error":"true"}},
				{"nested":{"path":"tags","query":{"bool":{"must":[
					{"term":{"tags.key":"error"}},
					{"term":{"tags.value":"true"}}]}}}},
				{"nested":{"path":"process.tags","query":{"bool":{"must":[
					{"term":{"process.tags.key":"error"}},
					{"term":{"process.tags.value":"true"}}]}}}}]}}}}`,
		},
		{
			name:   "a non-boolean error value keeps its literal match",
			filter: call(tracestore.OpEq, spanAttr("error"), scalar("oops")),
			want: `{"bool":{"should":[
				{"term":{"tag.error":"oops"}},
				{"nested":{"path":"tags","query":{"bool":{"must":[
					{"term":{"tags.key":"error"}},
					{"term":{"tags.value":"oops"}}]}}}}]}}`,
		},
		{
			name:   "an error tag at the resource level is an ordinary attribute",
			filter: call(tracestore.OpEq, resourceAttr("error"), scalar("false")),
			want: `{"bool":{"should":[
				{"term":{"process.tag.error":"false"}},
				{"nested":{"path":"process.tags","query":{"bool":{"must":[
					{"term":{"process.tags.key":"error"}},
					{"term":{"process.tags.value":"false"}}]}}}}]}}`,
		},
	}
	withSpanReader(t, func(r *spanReaderTest) {
		for _, test := range tests {
			t.Run(test.name, func(t *testing.T) {
				query, err := r.reader.buildFilterQuery(test.filter)
				require.NoError(t, err)
				source, err := query.Source()
				require.NoError(t, err)
				got, err := json.Marshal(source)
				require.NoError(t, err)
				assert.JSONEq(t, test.want, string(got))
			})
		}
	})
}

// TestBuildFilterQueryRefused covers what this schema cannot answer. Every case is refused
// rather than approximated, and each refusal is one of the two sentinels the API layers turn
// into a 400 — ErrFilterUnsupported for a predicate this backend cannot serve,
// ErrFilterInvalid for one that is malformed however it is served.
func TestBuildFilterQueryRefused(t *testing.T) {
	tests := []struct {
		name    string
		filter  *tracestore.Call
		wantErr error
		wantMsg string
	}{
		{
			name: "the instrumentation level is folded into the span's own tags",
			filter: call(tracestore.OpEq,
				&tracestore.Reference{Name: "otel.scope.name", Level: tracestore.LevelInstrumentation, Attr: true},
				scalar("lib")),
			wantErr: tracestore.ErrFilterUnsupported,
			wantMsg: `does not index the "instrumentation" level`,
		},
		{
			name: "link attributes are not indexed at all",
			filter: call(tracestore.OpEq,
				&tracestore.Reference{Name: "k", Level: tracestore.LevelLink, Attr: true},
				scalar("v")),
			wantErr: tracestore.ErrFilterUnsupported,
			wantMsg: `does not index the "link" level`,
		},
		{
			name: "a built-in field this schema has no field for",
			filter: call(tracestore.OpEq,
				&tracestore.Reference{Name: "kind", Level: tracestore.LevelSpan},
				scalar("server")),
			wantErr: tracestore.ErrFilterUnsupported,
			wantMsg: `built-in field "kind" of the "span" level`,
		},
		{
			name: "exists on a built-in field this schema has no field for",
			filter: call(tracestore.OpExists,
				&tracestore.Reference{Name: "traceID", Level: tracestore.LevelLink}),
			wantErr: tracestore.ErrFilterUnsupported,
			wantMsg: `built-in field "traceID" of the "link" level`,
		},
		{
			name:    "ordering an attribute, which is indexed as a keyword",
			filter:  call(tracestore.OpGt, spanAttr("http.response.size"), scalar("500")),
			wantErr: tracestore.ErrFilterUnsupported,
			wantMsg: `indexes "http.response.size" as a keyword rather than a number`,
		},
		{
			name:    "ordering the operation name",
			filter:  call(tracestore.OpLte, tracestore.SpanName.Ref(), scalar("m")),
			wantErr: tracestore.ErrFilterUnsupported,
			wantMsg: `indexes "name" as a keyword rather than a number`,
		},
		{
			name:    "a pattern over the duration, which is a number",
			filter:  call(tracestore.OpRegex, tracestore.SpanDuration.Ref(), scalar("2.*")),
			wantErr: tracestore.ErrFilterUnsupported,
			wantMsg: `operator "regex" on a duration`,
		},
		{
			name: "a constant that declares its type, which there is no typed storage to route to",
			filter: call(tracestore.OpEq, spanAttr("http.status_code"),
				&tracestore.Scalar{Value: "500", Type: tracestore.ValueTypeInt}),
			wantErr: tracestore.ErrFilterUnsupported,
			wantMsg: `declared as "int"`,
		},
		{
			name: "a typed list, whose type applies to every member",
			filter: call(tracestore.OpIn, spanAttr("http.status_code"),
				&tracestore.List{Values: []string{"500"}, Type: tracestore.ValueTypeInt}),
			wantErr: tracestore.ErrFilterUnsupported,
			wantMsg: `declared as "int"`,
		},
		{
			name: "the some quantifier, whose correlated matching is not implemented",
			filter: call(tracestore.OpSome,
				&tracestore.Reference{Level: tracestore.LevelEvent},
				call(tracestore.OpEq, tracestore.EventName.Ref(), scalar("exception"))),
			wantErr: tracestore.ErrFilterUnsupported,
			wantMsg: `operator "some"`,
		},
		{
			name:    "an operator this build does not know",
			filter:  call("json_extract", spanAttr("input"), scalar("$.a")),
			wantErr: tracestore.ErrFilterUnsupported,
			wantMsg: `operator "json_extract"`,
		},
		{
			name:    "comparing two values read off the same span, which would need a script",
			filter:  call(tracestore.OpNe, spanAttr("enduser.id"), resourceAttr("enduser.id")),
			wantErr: tracestore.ErrFilterUnsupported,
			wantMsg: `evaluates "ne" against a constant only`,
		},
		{
			name: "membership against something other than a list",
			filter: call(tracestore.OpIn, tracestore.ResourceService.Ref(),
				scalar("cart")),
			wantErr: tracestore.ErrFilterUnsupported,
			wantMsg: `evaluates "in" against a constant only`,
		},
		{
			name:    "a duration value with no unit",
			filter:  call(tracestore.OpGt, tracestore.SpanDuration.Ref(), scalar("2")),
			wantErr: tracestore.ErrFilterInvalid,
			wantMsg: `"2" is not a duration such as "2s"`,
		},
		{
			// ne builds the presence test before the comparison, so this reaches the second of
			// the two and is refused by it.
			name:    "a duration value with no unit, negated",
			filter:  call(tracestore.OpNe, tracestore.SpanDuration.Ref(), scalar("2")),
			wantErr: tracestore.ErrFilterInvalid,
			wantMsg: `"2" is not a duration such as "2s"`,
		},
		{
			name: "a list member that is not a duration",
			filter: call(tracestore.OpNotIn, tracestore.SpanDuration.Ref(),
				&tracestore.List{Values: []string{"2s", "later"}}),
			wantErr: tracestore.ErrFilterInvalid,
			wantMsg: `"later" is not a duration such as "2s"`,
		},
		{
			name:    "a comparison with the wrong number of arguments",
			filter:  call(tracestore.OpEq, spanAttr("k")),
			wantErr: tracestore.ErrFilterInvalid,
			wantMsg: `"eq" cannot take 1 arguments`,
		},
		{
			name: "not, which negates exactly one predicate",
			filter: call(tracestore.OpNot,
				call(tracestore.OpEq, spanAttr("a"), scalar("1")),
				call(tracestore.OpEq, spanAttr("b"), scalar("2"))),
			wantErr: tracestore.ErrFilterInvalid,
			wantMsg: `"not" cannot take 2 arguments`,
		},
		{
			name:    "a combinator with no arguments",
			filter:  call(tracestore.OpAnd),
			wantErr: tracestore.ErrFilterInvalid,
			wantMsg: `"and" cannot take 0 arguments`,
		},
		{
			name:    "a combinator given a value where a predicate belongs",
			filter:  call(tracestore.OpAnd, spanAttr("k"), scalar("v")),
			wantErr: tracestore.ErrFilterInvalid,
			wantMsg: `"and" combines predicates, not values`,
		},
		{
			name:    "exists given a constant, which reads nothing off the span",
			filter:  call(tracestore.OpExists, scalar("k")),
			wantErr: tracestore.ErrFilterInvalid,
			wantMsg: `"exists" reads a value on the span`,
		},
		{
			name:    "exists with the wrong number of arguments",
			filter:  call(tracestore.OpExists, spanAttr("a"), spanAttr("b")),
			wantErr: tracestore.ErrFilterInvalid,
			wantMsg: `"exists" cannot take 2 arguments`,
		},
		{
			name: "membership against an empty list, which nothing can satisfy",
			filter: call(tracestore.OpIn, tracestore.ResourceService.Ref(),
				&tracestore.List{}),
			wantErr: tracestore.ErrFilterInvalid,
			wantMsg: `"in" compares against an empty list`,
		},
		{
			name: "membership with the wrong number of arguments",
			filter: call(tracestore.OpNotIn,
				&tracestore.List{Values: []string{"a"}}),
			wantErr: tracestore.ErrFilterInvalid,
			wantMsg: `"not_in" cannot take 1 arguments`,
		},
	}
	withSpanReader(t, func(r *spanReaderTest) {
		for _, test := range tests {
			t.Run(test.name, func(t *testing.T) {
				query, err := r.reader.buildFilterQuery(test.filter)
				assert.Nil(t, query)
				require.ErrorIs(t, err, test.wantErr)
				assert.Contains(t, err.Error(), test.wantMsg)
			})
		}
	})
}

// TestBuildFindTraceIDsQueryWithFilter checks how the filter joins the search: as one more
// must clause beside the time range, which is the only other clause a filter query carries
// because the query service keeps the legacy predicate fields out of it.
func TestBuildFindTraceIDsQueryWithFilter(t *testing.T) {
	withSpanReader(t, func(r *spanReaderTest) {
		start := time.Time{}
		end := time.Time{}.Add(time.Second)
		query, err := r.reader.buildFindTraceIDsQuery(dbmodel.TraceQueryParameters{
			StartTimeMin: start,
			StartTimeMax: end,
			Filter:       call(tracestore.OpEq, tracestore.ResourceService.Ref(), scalar("cart")),
		})
		require.NoError(t, err)
		got, err := query.Source()
		require.NoError(t, err)
		want, err := esquery.NewBoolQuery().Must(
			r.reader.buildStartTimeQuery(start, end),
			esquery.NewTermQuery(serviceNameField, "cart"),
		).Source()
		require.NoError(t, err)
		assert.Equal(t, want, got)
	})
}

// TestFindTraceIDsRefusesUnservableFilter checks that a refusal reaches the caller instead of
// a search running without the predicate it could not lower, which would answer a different
// question than the one asked.
func TestFindTraceIDsRefusesUnservableFilter(t *testing.T) {
	withSpanReader(t, func(r *spanReaderTest) {
		now := time.Now()
		_, err := r.reader.FindTraceIDs(context.Background(), dbmodel.TraceQueryParameters{
			StartTimeMin: now,
			StartTimeMax: now.Add(time.Hour),
			Filter: call(tracestore.OpEq,
				&tracestore.Reference{Name: "kind", Level: tracestore.LevelSpan},
				scalar("server")),
		})
		require.ErrorIs(t, err, tracestore.ErrFilterUnsupported)
	})
}

// TestBuildFilterQueryRefusalFromWithin checks that a refusal deep in the tree is the
// refusal the caller sees, rather than being lost as a partially built query.
func TestBuildFilterQueryRefusalFromWithin(t *testing.T) {
	unservable := call(tracestore.OpEq,
		&tracestore.Reference{Name: "k", Level: tracestore.LevelLink, Attr: true},
		scalar("v"))
	servable := call(tracestore.OpEq, tracestore.ResourceService.Ref(), scalar("cart"))
	for _, filter := range []*tracestore.Call{
		call(tracestore.OpAnd, servable, unservable),
		call(tracestore.OpOr, servable, unservable),
		call(tracestore.OpNot, unservable),
		call(tracestore.OpAnd, servable, call(tracestore.OpOr, servable, unservable)),
		// The negated leaves build the positive comparison and the presence test in turn, so
		// either of them can be the one that refuses.
		call(tracestore.OpNe,
			&tracestore.Reference{Name: "k", Level: tracestore.LevelLink, Attr: true},
			scalar("v")),
		call(tracestore.OpNotIn,
			&tracestore.Reference{Name: "k", Level: tracestore.LevelLink, Attr: true},
			&tracestore.List{Values: []string{"v"}}),
	} {
		withSpanReader(t, func(r *spanReaderTest) {
			query, err := r.reader.buildFilterQuery(filter)
			assert.Nil(t, query)
			require.ErrorIs(t, err, tracestore.ErrFilterUnsupported)
			assert.Contains(t, err.Error(), `does not index the "link" level`)
		})
	}
}
