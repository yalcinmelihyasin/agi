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

const mipLevels = 100

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
	for _, cmd := range inputCommands {
		switch typedCmd := cmd.(type) {
		case *VkCreateSampler:
			outputCmds = append(outputCmds, experiment.modifySamplerCreation(ctx, id, typedCmd, inputState))
		case *VkCreateImage:
			outputCmds = append(outputCmds, experiment.modifyImageCreation(ctx, id, typedCmd, inputState))
		case *VkCreateImageView:
			outputCmds = append(outputCmds, experiment.modifyImageViewCreation(ctx, id, typedCmd, inputState))
		case *VkCmdPipelineBarrier:
			outputCmds = append(outputCmds, experiment.modifyCmdPipelineBarrier(ctx, id, typedCmd, inputState))
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
	if !experiment.generateMipmaps && !experiment.disableAF {
		return cmd
	}

	cmd.Extras().Observations().ApplyReads(inputState.Memory.ApplicationPool())

	pAlloc := memory.Pointer(cmd.PAllocator())
	pSampler := memory.Pointer(cmd.PSampler())

	pInfo := cmd.PCreateInfo()
	info := pInfo.MustRead(ctx, cmd, inputState, nil)

	if experiment.disableAF {
		if info.AnisotropyEnable() != VkBool32(0) {
			log.I(ctx, "Anisotropy disabled for VKCreateSampler[%v]", id)
			info.SetAnisotropyEnable(VkBool32(0))
			info.SetMaxAnisotropy(0.0)
		}
	}

	// Melih TODO: How To check PNext type?
	// We have to check if VkSamplerYcbcrConversionCreateInfo exists.

	if experiment.generateMipmaps {
		// Melih TODO: We should reliable check here if mipmapping is enabled
		// and possibly do nothing if it's already enabled
		if info.UnnormalizedCoordinates() != VkBool32(0) {
			info.SetMipmapMode(VkSamplerMipmapMode_VK_SAMPLER_MIPMAP_MODE_NEAREST)
			info.SetMinLod(0.0)
			info.SetMaxLod(0.0)
			info.SetMinFilter(VkFilter_VK_FILTER_NEAREST)
			info.SetMagFilter(VkFilter_VK_FILTER_NEAREST)
		} else {
			info.SetMipmapMode(VkSamplerMipmapMode_VK_SAMPLER_MIPMAP_MODE_NEAREST)
			info.SetMinLod(0.0)
			info.SetMaxLod(mipLevels)
			info.SetMinFilter(VkFilter_VK_FILTER_NEAREST)
			info.SetMagFilter(VkFilter_VK_FILTER_NEAREST)
		}

		log.I(ctx, "Mipmaps set for sampler at cmd %v", id)
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

func (experiment *textureExperiments) modifyImageCreation(ctx context.Context, id transform.CommandID, cmd *VkCreateImage, inputState *api.GlobalState) api.Cmd {
	if !experiment.generateMipmaps {
		return cmd
	}

	cmd.Extras().Observations().ApplyReads(inputState.Memory.ApplicationPool())

	pAlloc := memory.Pointer(cmd.PAllocator())
	pImage := memory.Pointer(cmd.PImage())

	pInfo := cmd.PCreateInfo()
	info := pInfo.MustRead(ctx, cmd, inputState, nil)

	if info.MipLevels() != 0 {
		// Mipmaps are already enabled
		// Melih TODO: Is this safe as reads already happened?
		return cmd
	}
	info.SetMipLevels(mipLevels)
	newInfo := experiment.allocations.AllocDataOrPanic(ctx, info)

	cb := CommandBuilder{Thread: cmd.Thread(), Arena: inputState.Arena}
	newCmd := cb.VkCreateImage(cmd.Device(), newInfo.Ptr(), pAlloc, pImage, cmd.Result())
	newCmd.AddRead(newInfo.Data())

	for _, w := range cmd.Extras().Observations().Writes {
		newCmd.AddWrite(w.Range, w.ID)
	}

	log.I(ctx, "Mipmaps set for Image at cmd %v", id)
	return cmd
}

func (experiment *textureExperiments) modifyImageViewCreation(ctx context.Context, id transform.CommandID, cmd *VkCreateImageView, inputState *api.GlobalState) api.Cmd {
	if !experiment.generateMipmaps {
		return cmd
	}

	cmd.Extras().Observations().ApplyReads(inputState.Memory.ApplicationPool())

	pAlloc := memory.Pointer(cmd.PAllocator())
	pView := memory.Pointer(cmd.PView())

	pInfo := cmd.PCreateInfo()
	info := pInfo.MustRead(ctx, cmd, inputState, nil)

	if info.SubresourceRange().BaseMipLevel() != 0 || info.SubresourceRange().LevelCount() != 0 {
		// ImageView already has mipmaps
		// Melih TODO: Is this safe as reads already happened?
		return cmd
	}

	// Melih TODO: This is not a pointer therefore can be directly manipulated right?
	info.SubresourceRange().SetBaseMipLevel(0)
	info.SubresourceRange().SetLevelCount(mipLevels)
	newInfo := experiment.allocations.AllocDataOrPanic(ctx, info)

	cb := CommandBuilder{Thread: cmd.Thread(), Arena: inputState.Arena}
	newCmd := cb.VkCreateImageView(cmd.Device(), newInfo.Ptr(), pAlloc, pView, cmd.Result())
	newCmd.AddRead(newInfo.Data())

	for _, w := range cmd.Extras().Observations().Writes {
		newCmd.AddWrite(w.Range, w.ID)
	}

	log.I(ctx, "Mipmaps set for ImageView at cmd %v", id)
	return cmd
}

func (experiment *textureExperiments) modifyCmdPipelineBarrier(ctx context.Context, id transform.CommandID, cmd *VkCmdPipelineBarrier, inputState *api.GlobalState) api.Cmd {
	if !experiment.generateMipmaps {
		return cmd
	}

	cmd.Extras().Observations().ApplyReads(inputState.Memory.ApplicationPool())
	imageMemoryBarriers := cmd.PImageMemoryBarriers().
		Slice(0, uint64(cmd.ImageMemoryBarrierCount()), inputState.MemoryLayout).
		MustRead(ctx, cmd, inputState, nil)

	newBarriers := make([]VkImageMemoryBarrier, 0, cmd.ImageMemoryBarrierCount())
	for _, barrier := range imageMemoryBarriers {
		// Melih TODO: This is not a pointer therefore can be directly manipulated right?
		if barrier.SubresourceRange().BaseMipLevel() != 0 || barrier.SubresourceRange().LevelCount() != 0 {
			newBarriers = append(newBarriers, barrier)
			continue
		}

		newBarrier := barrier.Clone(inputState.Arena, api.CloneContext{})
		// Melih TODO: This is not a pointer therefore can be directly manipulated right?
		newBarrier.SubresourceRange().SetBaseMipLevel(0)
		newBarrier.SubresourceRange().SetLevelCount(mipLevels)

		newBarriers = append(newBarriers, newBarrier)
	}

	newBarriersAlloc := experiment.allocations.AllocDataOrPanic(ctx, newBarriers)
	// Melih TODO: Is this necessary(copied from MustAllocData in command splitter)
	rng, memId := newBarriersAlloc.Data()
	inputState.Memory.ApplicationPool().Write(rng.Base, memory.Resource(memId, rng.Size))

	cb := CommandBuilder{Thread: cmd.Thread(), Arena: inputState.Arena}
	newCmd := cb.VkCmdPipelineBarrier(
		cmd.CommandBuffer(),
		cmd.SrcStageMask(),
		cmd.DstStageMask(),
		cmd.DependencyFlags(),
		cmd.MemoryBarrierCount(),
		memory.Pointer(cmd.PMemoryBarriers()),
		cmd.BufferMemoryBarrierCount(),
		memory.Pointer(cmd.PBufferMemoryBarriers()),
		cmd.ImageMemoryBarrierCount(),
		NewVkImageMemoryBarrierᶜᵖ(newBarriersAlloc.Ptr()),
	)
	// Melih TODO: Should we clone this? If we should, when?
	newCmd.Extras().MustClone(cmd.Extras().All()...)
	newCmd.AddRead(newBarriersAlloc.Data())

	log.I(ctx, "Mipmaps set for pipeline barrier at cmd %v", id)
	return newCmd
}
