// Copyright (C) 2017 Google Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package log

import (
	"context"

	"github.com/google/agi/core/context/keys"
)

type processKeyTy string

const processKey processKeyTy = "log.processKey"

// PutProcess returns a new context with the process assigned to w.
func PutProcess(ctx context.Context, w string) context.Context {
	return keys.WithValue(ctx, processKey, w)
}

// GetProcess returns the process assigned to ctx.
func GetProcess(ctx context.Context) string {
	out, _ := ctx.Value(processKey).(string)
	return out
}
