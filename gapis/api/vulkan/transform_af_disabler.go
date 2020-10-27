// Copyright (C) 2020 Google Inc.
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

package vulkan

import (
	"context"

	"github.com/google/gapid/core/log"
	"github.com/google/gapid/gapis/api"
	"github.com/google/gapid/gapis/api/transform"
	"github.com/google/gapid/gapis/memory"
)

type textureExperiments struct {
	allocations     *allocationTracker
	disableAF       bool
	generateMipmaps bool
}

func newTextureExperimentsTransform(disableAF bool, generateMipmaps bool) *textureExperiments {
	return &textureExperiments{
		allocations:     nil,
		disableAF:       disableAF,
		generateMipmaps: generateMipmaps,
	}
}

func (experiment *textureExperiments) RequiresAccurateState() bool {
	return false
}

func (experiment *textureExperiments) RequiresInnerStateMutation() bool {
	return false
}

func (experiment *textureExperiments) SetInnerStateMutationFunction(mutator transform.StateMutator) {
	// This transform does not require inner state mutation
}

func (experiment *textureExperiments) BeginTransform(ctx context.Context, inputState *api.GlobalState) error {
	experiment.allocations = NewAllocationTracker(inputState)
	return nil
}

func (experiment *textureExperiments) ClearTransformResources(ctx context.Context) {
	experiment.allocations.FreeAllocations()
}

func (experiment *textureExperiments) TransformCommand(ctx context.Context, id transform.CommandID, inputCommands []api.Cmd, inputState *api.GlobalState) ([]api.Cmd, error) {
	outputCmds := make([]api.Cmd, 0, len(inputCommands))
	for i, cmd := range inputCommands {
		switch typedCmd := cmd.(type) {
		case *VkCreateSampler:
			outputCmds = append(outputCmds, experiment.modifySamplerCreation(ctx, id, typedCmd, inputState))
		case *VkCreateImage:
			outputCmds = append(outputCmds, experiment.modifyImageCreation(ctx, id, typedCmd, inputState)...)
		default:
			outputCmds = append(outputCmds, cmd)
		}
	}

	return outputCmds, nil
}

func (experiment *textureExperiments) EndTransform(ctx context.Context, inputState *api.GlobalState) ([]api.Cmd, error) {
	return nil, nil
}

func (experiment *textureExperiments) modifySamplerCreation(ctx context.Context, id transform.CommandID, cmd *VkCreateSampler, inputState *api.GlobalState) api.Cmd {
	cmd.Extras().Observations().ApplyReads(inputState.Memory.ApplicationPool())

	pAlloc := memory.Pointer(cmd.PAllocator())
	pSampler := memory.Pointer(cmd.PSampler())

	pInfo := cmd.PCreateInfo()
	info := pInfo.MustRead(ctx, cmd, inputState, nil)

	if experiment.disableAF {
		log.I(ctx, "Anisotropy disabled for VKCreateSampler[%v]", id)
		info.SetAnisotropyEnable(VkBool32(0))
	}

	// Melih TODO: How To check PNext type?
	// We have to check if VkSamplerYcbcrConversionCreateInfo exists.

	if experiment.generateMipmaps {
		if info.UnnormalizedCoordinates() != VkBool32(0) {
			info.SetMipmapMode(VkSamplerMipmapMode_VK_SAMPLER_MIPMAP_MODE_NEAREST)
			info.SetMinLod(0.0)
			info.SetMaxLod(0.0)
			info.SetMinFilter(VkFilter_VK_FILTER_NEAREST)
			info.SetMagFilter(VkFilter_VK_FILTER_NEAREST)
		} else {
			info.SetMipmapMode(VkSamplerMipmapMode_VK_SAMPLER_MIPMAP_MODE_NEAREST)
			info.SetMinLod(0.0)
			info.SetMaxLod(0.25)
			info.SetMinFilter(VkFilter_VK_FILTER_NEAREST)
			info.SetMagFilter(VkFilter_VK_FILTER_LINEAR)
		}
	}

	newInfo := experiment.allocations.AllocDataOrPanic(ctx, info)

	cb := CommandBuilder{Thread: cmd.Thread(), Arena: inputState.Arena}
	newCmd := cb.VkCreateSampler(cmd.Device(), newInfo.Ptr(), pAlloc, pSampler, cmd.Result())
	newCmd.AddRead(newInfo.Data())
	for _, w := range cmd.Extras().Observations().Writes {
		newCmd.AddWrite(w.Range, w.ID)
	}

	return cmd
}

func (experiment *textureExperiments) modifyImageCreation(ctx context.Context, id transform.CommandID, cmd *VkCreateImage, inputState *api.GlobalState) []api.Cmd {
	cmd.Extras().Observations().ApplyReads(inputState.Memory.ApplicationPool())
	pInfo := cmd.PCreateInfo()
	info := pInfo.MustRead(ctx, cmd, inputState, nil)

	width := info.Extent().Width()
	height := info.Extent().Height()

	return []api.Cmd{cmd}
}
