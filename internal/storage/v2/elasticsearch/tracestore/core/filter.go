// Copyright (c) 2026 The Jaeger Authors.
// SPDX-License-Identifier: Apache-2.0

package core

import (
	"fmt"
	"strconv"
	"time"

	"github.com/jaegertracing/jaeger-idl/model/v1"
	esquery "github.com/jaegertracing/jaeger/internal/storage/elasticsearch/query"
	"github.com/jaegertracing/jaeger/internal/storage/v2/api/tracestore"
)

// eventNameKey is the logs.fields key the write path stores a span event's name under
// (spanEventsToDbSpanLogs), which is what makes the event.name built-in field readable.
const eventNameKey = "event"

// eventNameAsAttribute reads event.name as the logs.fields entry it is stored in, so the
// built-in field and an event attribute share one lowering.
var eventNameAsAttribute = &tracestore.Reference{
	Name:  eventNameKey,
	Level: tracestore.LevelEvent,
	Attr:  true,
}

// attributeLocations is where a level's attributes live in a span document: the flattened
// object fields, whose leaf is the attribute key, and the nested key/value arrays. Both are
// searched because which of the two the write path produced depends on the tags-as-fields
// setting in force when the span was indexed, and that setting can change over the life of
// an index. Instrumentation-scope attributes are folded into the span's own tags and link
// attributes are not indexed at all, so neither level appears here (RFC 0005 §1.6).
var attributeLocations = map[tracestore.Level]attributeLocation{
	tracestore.LevelSpan: {
		object: []string{objectTagsField},
		nested: []string{nestedTagsField},
	},
	tracestore.LevelResource: {
		object: []string{objectProcessTagsField},
		nested: []string{nestedProcessTagsField},
	},
	tracestore.LevelEvent: {
		nested: []string{nestedLogFieldsField},
	},
	// An unqualified reference searches the span and resource levels (RFC 0005 §5.1). It
	// deliberately stops short of the event level that the legacy Tags search also covers:
	// the legacy field keeps its behavior, and the filter follows the documented contract.
	"": {
		object: objectTagFieldList,
		nested: []string{nestedTagsField, nestedProcessTagsField},
	},
}

type attributeLocation struct {
	object []string
	nested []string
}

// valueMatch builds the query matching a value held in one Elasticsearch field. An
// attribute lives in several fields at once, so the comparison is chosen from the operator
// once and then applied to each of them.
type valueMatch func(field string) esquery.Query

// FilterCapabilities declares the part of the RFC 0005 filter model this reader evaluates.
// It omits the instrumentation and link levels, which the schema does not index separately,
// and the `some` quantifier, whose correlated matching over a span's events is not
// implemented yet. Which built-in fields are served is not declarable — a field name is
// indistinguishable from an attribute key — so buildFilterQuery refuses the ones this
// schema has no field for.
func FilterCapabilities() tracestore.FilterCapabilities {
	return tracestore.FilterCapabilities{
		Levels: []tracestore.Level{
			tracestore.LevelSpan,
			tracestore.LevelResource,
			tracestore.LevelEvent,
		},
		Operators: []tracestore.Operator{
			tracestore.OpAnd,
			tracestore.OpOr,
			tracestore.OpNot,
			tracestore.OpEq,
			tracestore.OpNe,
			tracestore.OpGt,
			tracestore.OpLt,
			tracestore.OpGte,
			tracestore.OpLte,
			tracestore.OpRegex,
			tracestore.OpExists,
			tracestore.OpIn,
			tracestore.OpNotIn,
		},
	}
}

// buildFilterQuery lowers a structured query filter (RFC 0005) into an Elasticsearch query.
// The boolean combinators become the bool query's must / should / must_not clauses, and a
// leaf comparison becomes a term, regexp, or range query over the fields the referenced
// value lives in. A predicate this schema cannot answer is refused rather than approximated,
// so a caller never reads a narrower answer as the whole one.
//
// The filter arrives already checked against FilterCapabilities when it comes through the
// query service, but a remote-storage client can reach this reader without that check, so
// every refusal is made here too rather than assumed.
func (s *SpanReader) buildFilterQuery(predicate *tracestore.Call) (esquery.Query, error) {
	switch predicate.Op {
	case tracestore.OpAnd, tracestore.OpOr:
		args, err := s.buildCombinedArgs(predicate)
		if err != nil {
			return nil, err
		}
		if predicate.Op == tracestore.OpAnd {
			return allOf(args), nil
		}
		return anyOf(args), nil

	case tracestore.OpNot:
		if len(predicate.Args) != 1 {
			return nil, errArity(predicate)
		}
		args, err := s.buildCombinedArgs(predicate)
		if err != nil {
			return nil, err
		}
		return esquery.NewBoolQuery().MustNot(args[0]), nil

	case tracestore.OpEq, tracestore.OpRegex,
		tracestore.OpGt, tracestore.OpLt, tracestore.OpGte, tracestore.OpLte:
		ref, value, err := refAndScalarArgs(predicate)
		if err != nil {
			return nil, err
		}
		return s.buildComparison(predicate.Op, ref, value)

	case tracestore.OpNe:
		ref, value, err := refAndScalarArgs(predicate)
		if err != nil {
			return nil, err
		}
		present, err := s.buildExists(ref)
		if err != nil {
			return nil, err
		}
		equal, err := s.buildComparison(tracestore.OpEq, ref, value)
		if err != nil {
			return nil, err
		}
		return holdsSomethingElse(present, equal), nil

	case tracestore.OpIn, tracestore.OpNotIn:
		return s.buildMembership(predicate)

	case tracestore.OpExists:
		ref, err := refArg(predicate)
		if err != nil {
			return nil, err
		}
		return s.buildExists(ref)

	default:
		return nil, fmt.Errorf("%w: it does not support the operator %q",
			tracestore.ErrFilterUnsupported, predicate.Op)
	}
}

// buildCombinedArgs lowers the arguments of a boolean combinator, each of which is itself a
// predicate rather than a value.
func (s *SpanReader) buildCombinedArgs(predicate *tracestore.Call) ([]esquery.Query, error) {
	if len(predicate.Args) == 0 {
		return nil, errArity(predicate)
	}
	queries := make([]esquery.Query, 0, len(predicate.Args))
	for _, arg := range predicate.Args {
		call, ok := arg.(*tracestore.Call)
		if !ok {
			return nil, fmt.Errorf("%w: %q combines predicates, not values",
				tracestore.ErrFilterInvalid, predicate.Op)
		}
		query, err := s.buildFilterQuery(call)
		if err != nil {
			return nil, err
		}
		queries = append(queries, query)
	}
	return queries, nil
}

// buildMembership lowers `in` and `not_in` as the disjunction of equalities they stand for,
// so membership reaches every reference an equality does without a second lowering.
func (s *SpanReader) buildMembership(predicate *tracestore.Call) (esquery.Query, error) {
	ref, list, err := refAndListArgs(predicate)
	if err != nil {
		return nil, err
	}
	if len(list.Values) == 0 {
		return nil, fmt.Errorf("%w: %q compares against an empty list",
			tracestore.ErrFilterInvalid, predicate.Op)
	}
	// The presence test is built even for `in`, which does not need it, so that a reference
	// this reader cannot read is refused before the members are lowered one by one.
	present, err := s.buildExists(ref)
	if err != nil {
		return nil, err
	}
	members := make([]esquery.Query, 0, len(list.Values))
	for _, value := range list.Values {
		member, err := s.buildComparison(tracestore.OpEq, ref, &tracestore.Scalar{Value: value, Type: list.Type})
		if err != nil {
			return nil, err
		}
		members = append(members, member)
	}
	if predicate.Op == tracestore.OpIn {
		return anyOf(members), nil
	}
	return holdsSomethingElse(present, anyOf(members)), nil
}

// holdsSomethingElse builds the negated leaf comparisons, `ne` and `not_in`. A comparison
// against a value the span does not carry is false, and only a boolean `not` flips that
// (RFC 0005 §5.3), so these ask for the spans that hold the reference and hold something
// else — not merely the ones a bare must_not would leave, which include every span missing
// the reference entirely.
func holdsSomethingElse(present, matching esquery.Query) esquery.Query {
	return esquery.NewBoolQuery().Must(present).MustNot(matching)
}

func (s *SpanReader) buildComparison(
	op tracestore.Operator,
	ref *tracestore.Reference,
	value *tracestore.Scalar,
) (esquery.Query, error) {
	if value.Type != "" {
		return nil, errTypedConstant(value.Type)
	}
	switch {
	case ref.IsAttribute():
		return s.buildAttributeComparison(op, ref, value.Value)
	case ref.IsField(tracestore.SpanName):
		return buildTextComparison(operationNameField, op, ref, value.Value)
	case ref.IsField(tracestore.ResourceService):
		return buildTextComparison(serviceNameField, op, ref, value.Value)
	case ref.IsField(tracestore.SpanDuration):
		return buildDurationComparison(op, value.Value)
	case ref.IsField(tracestore.EventName):
		return s.buildAttributeComparison(op, eventNameAsAttribute, value.Value)
	default:
		return nil, errUnsupportedField(ref)
	}
}

func (s *SpanReader) buildExists(ref *tracestore.Reference) (esquery.Query, error) {
	switch {
	case ref.IsAttribute():
		return s.buildAttributeExists(ref)
	case ref.IsField(tracestore.SpanName):
		return esquery.NewExistsQuery(operationNameField), nil
	case ref.IsField(tracestore.ResourceService):
		return esquery.NewExistsQuery(serviceNameField), nil
	case ref.IsField(tracestore.SpanDuration):
		return esquery.NewExistsQuery(durationField), nil
	case ref.IsField(tracestore.EventName):
		return s.buildAttributeExists(eventNameAsAttribute)
	default:
		return nil, errUnsupportedField(ref)
	}
}

func (s *SpanReader) buildAttributeComparison(
	op tracestore.Operator,
	ref *tracestore.Reference,
	value string,
) (esquery.Query, error) {
	locations, ok := attributeLocations[ref.Level]
	if !ok {
		return nil, errUnsupportedLevel(ref.Level)
	}
	if isError, ok := asErrorTagEquality(op, ref, value); ok {
		// The write path records the error tag only for a span whose status is an error
		// (getTagFromStatusCode), so a span that succeeded carries no error tag and matching
		// error=false literally returns nothing. Read it as the complement — every span that
		// is not an error — as the legacy tag search and the in-memory store both do (#9096).
		errored := s.attributeQuery(locations, ref.Name, termMatch("true"))
		if isError {
			return errored, nil
		}
		return esquery.NewBoolQuery().MustNot(errored), nil
	}
	match, err := attributeValueMatch(op, ref, value)
	if err != nil {
		return nil, err
	}
	return s.attributeQuery(locations, ref.Name, match), nil
}

// attributeQuery matches an attribute in every field its level keeps attributes in.
func (s *SpanReader) attributeQuery(locations attributeLocation, key string, match valueMatch) esquery.Query {
	queries := make([]esquery.Query, 0, len(locations.object)+len(locations.nested))
	for _, field := range locations.object {
		queries = append(queries, match(s.objectAttributeField(field, key)))
	}
	for _, path := range locations.nested {
		queries = append(queries, esquery.NewNestedQuery(path, esquery.NewBoolQuery().Must(
			esquery.NewTermQuery(nestedField(path, tagKeyField), key),
			match(nestedField(path, tagValueField)),
		)))
	}
	return anyOf(queries)
}

func (s *SpanReader) buildAttributeExists(ref *tracestore.Reference) (esquery.Query, error) {
	locations, ok := attributeLocations[ref.Level]
	if !ok {
		return nil, errUnsupportedLevel(ref.Level)
	}
	queries := make([]esquery.Query, 0, len(locations.object)+len(locations.nested))
	for _, field := range locations.object {
		queries = append(queries, esquery.NewExistsQuery(s.objectAttributeField(field, ref.Name)))
	}
	for _, path := range locations.nested {
		queries = append(queries, esquery.NewNestedQuery(path,
			esquery.NewTermQuery(nestedField(path, tagKeyField), ref.Name)))
	}
	return anyOf(queries), nil
}

// objectAttributeField is where the flattened representation keeps one attribute. Its leaf
// is the attribute key with dots replaced, because a dot in a field name would read as a
// step into a subobject.
func (s *SpanReader) objectAttributeField(field, key string) string {
	return field + "." + s.dotReplacer.ReplaceDot(key)
}

func nestedField(path, field string) string {
	return path + "." + field
}

// attributeValueMatch chooses how a comparison tests an attribute value. Attribute values
// are indexed as keywords, so equality and patterns work and ordering does not.
func attributeValueMatch(op tracestore.Operator, ref *tracestore.Reference, value string) (valueMatch, error) {
	switch op {
	case tracestore.OpEq:
		return termMatch(value), nil
	case tracestore.OpRegex:
		return func(field string) esquery.Query { return esquery.NewRegexpQuery(field, value) }, nil
	default:
		return nil, errUnorderedValue(op, ref)
	}
}

func termMatch(value string) valueMatch {
	return func(field string) esquery.Query { return esquery.NewTermQuery(field, value) }
}

// buildTextComparison compares a built-in field held as a keyword — an operation name or a
// service name — which supports equality and patterns but carries no order worth exposing.
func buildTextComparison(
	field string,
	op tracestore.Operator,
	ref *tracestore.Reference,
	value string,
) (esquery.Query, error) {
	switch op {
	case tracestore.OpEq:
		return esquery.NewTermQuery(field, value), nil
	case tracestore.OpRegex:
		return esquery.NewRegexpQuery(field, value), nil
	default:
		return nil, errUnorderedValue(op, ref)
	}
}

// durationComparisons is how each operator tests the duration field, which is the one
// ordered value this schema indexes numerically.
var durationComparisons = map[tracestore.Operator]func(micros uint64) esquery.Query{
	tracestore.OpEq: func(micros uint64) esquery.Query {
		return esquery.NewTermQuery(durationField, micros)
	},
	tracestore.OpGt: func(micros uint64) esquery.Query {
		return esquery.NewRangeQuery(durationField).Gt(micros)
	},
	tracestore.OpGte: func(micros uint64) esquery.Query {
		return esquery.NewRangeQuery(durationField).Gte(micros)
	},
	tracestore.OpLt: func(micros uint64) esquery.Query {
		return esquery.NewRangeQuery(durationField).Lt(micros)
	},
	tracestore.OpLte: func(micros uint64) esquery.Query {
		return esquery.NewRangeQuery(durationField).Lte(micros)
	},
}

// buildDurationComparison compares the span duration. The value carries its unit in Go
// duration syntax (RFC 0005 §5.3) and the field holds microseconds. The operator is resolved
// before the value is parsed, so an operator the duration has no answer for is refused as
// that rather than as a value that does not parse.
func buildDurationComparison(op tracestore.Operator, value string) (esquery.Query, error) {
	compare, ok := durationComparisons[op]
	if !ok {
		return nil, fmt.Errorf("%w: it does not support the operator %q on a duration",
			tracestore.ErrFilterUnsupported, op)
	}
	duration, err := time.ParseDuration(value)
	if err != nil {
		return nil, fmt.Errorf(`%w: %q is not a duration such as "2s"`, tracestore.ErrFilterInvalid, value)
	}
	return compare(model.DurationAsMicroseconds(duration)), nil
}

// asErrorTagEquality reports through ok whether a predicate tests the error tag for a
// boolean, the one attribute whose absence carries meaning, and through isError which side of
// it was asked for. The tag is written on the span, so a resource-level reference is not it.
func asErrorTagEquality(
	op tracestore.Operator,
	ref *tracestore.Reference,
	value string,
) (isError bool, ok bool) {
	if op != tracestore.OpEq || ref.Name != errorTag {
		return false, false
	}
	if ref.Level != "" && ref.Level != tracestore.LevelSpan {
		return false, false
	}
	isError, err := strconv.ParseBool(value)
	if err != nil {
		return false, false
	}
	return isError, true
}

// allOf and anyOf combine clauses, returning the one clause there is rather than wrapping a
// single alternative in a bool query that would mean the same thing.
func allOf(queries []esquery.Query) esquery.Query {
	if len(queries) == 1 {
		return queries[0]
	}
	return esquery.NewBoolQuery().Must(queries...)
}

func anyOf(queries []esquery.Query) esquery.Query {
	if len(queries) == 1 {
		return queries[0]
	}
	return esquery.NewBoolQuery().Should(queries...)
}

func refArg(predicate *tracestore.Call) (*tracestore.Reference, error) {
	if len(predicate.Args) != 1 {
		return nil, errArity(predicate)
	}
	ref, ok := predicate.Args[0].(*tracestore.Reference)
	if !ok {
		return nil, fmt.Errorf("%w: %q reads a value on the span",
			tracestore.ErrFilterInvalid, predicate.Op)
	}
	return ref, nil
}

func refAndScalarArgs(predicate *tracestore.Call) (*tracestore.Reference, *tracestore.Scalar, error) {
	if len(predicate.Args) != 2 {
		return nil, nil, errArity(predicate)
	}
	ref, refOK := predicate.Args[0].(*tracestore.Reference)
	value, valueOK := predicate.Args[1].(*tracestore.Scalar)
	if !refOK || !valueOK {
		return nil, nil, errRefAgainstConstant(predicate)
	}
	return ref, value, nil
}

func refAndListArgs(predicate *tracestore.Call) (*tracestore.Reference, *tracestore.List, error) {
	if len(predicate.Args) != 2 {
		return nil, nil, errArity(predicate)
	}
	ref, refOK := predicate.Args[0].(*tracestore.Reference)
	list, listOK := predicate.Args[1].(*tracestore.List)
	if !refOK || !listOK {
		return nil, nil, errRefAgainstConstant(predicate)
	}
	return ref, list, nil
}

func errArity(predicate *tracestore.Call) error {
	return fmt.Errorf("%w: %q cannot take %d arguments",
		tracestore.ErrFilterInvalid, predicate.Op, len(predicate.Args))
}

// errRefAgainstConstant refuses an operand shape the query engine has no expression for.
// Comparing two values read off the same span needs a script, which is too costly to run
// over every candidate span to offer here.
func errRefAgainstConstant(predicate *tracestore.Call) error {
	return fmt.Errorf("%w: it evaluates %q against a constant only",
		tracestore.ErrFilterUnsupported, predicate.Op)
}

func errUnsupportedLevel(level tracestore.Level) error {
	return fmt.Errorf("%w: it does not index the %q level", tracestore.ErrFilterUnsupported, level)
}

func errUnsupportedField(ref *tracestore.Reference) error {
	return fmt.Errorf("%w: it does not support the built-in field %q of the %q level",
		tracestore.ErrFilterUnsupported, ref.Name, ref.Level)
}

func errUnorderedValue(op tracestore.Operator, ref *tracestore.Reference) error {
	return fmt.Errorf("%w: it indexes %q as a keyword rather than a number, so it cannot evaluate %q on it",
		tracestore.ErrFilterUnsupported, ref.Name, op)
}

// errTypedConstant refuses a constant that declares its type. A declared type is
// authoritative — the backend must match only values stored at that type (RFC 0005 §5.4) —
// and the flattened attribute representation records no type to route by, so honoring it
// would silently mean matching more than was asked.
func errTypedConstant(valueType tracestore.ValueType) error {
	return fmt.Errorf("%w: it cannot route a constant declared as %q to a typed value, so it serves untyped constants only",
		tracestore.ErrFilterUnsupported, valueType)
}
