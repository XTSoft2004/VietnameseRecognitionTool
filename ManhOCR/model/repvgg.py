import torch.nn as nn

from ..model.repvgg_block import RepVGGBlock


class RepVGG(nn.Module):
    def __init__(
        self, num_blocks, width_multiplier=None, override_groups_map=None, deploy=False
    ):
        super(RepVGG, self).__init__()

        assert len(width_multiplier) == 4

        self.deploy = deploy
        self.override_groups_map = override_groups_map or dict()
        assert 0 not in self.override_groups_map

        self.in_planes = min(64, int(64 * width_multiplier[0]))

        self.stage0 = RepVGGBlock(
            in_channels=3,
            out_channels=self.in_planes,
            kernel_size=3,
            stride=2,
            padding=1,
            deploy=self.deploy,
        )
        self.cur_layer_idx = 1
        self.stage1 = self._make_stage(
            int(48 * width_multiplier[0]), num_blocks[0], stride=(2, 1)
        )
        self.stage2 = self._make_stage(
            int(96 * width_multiplier[1]), num_blocks[1], stride=(2, 1)
        )
        self.stage3 = self._make_stage(
            int(192 * width_multiplier[2]), num_blocks[2], stride=(2, 2)
        )
        self.stage4 = self._make_stage(
            int(384 * width_multiplier[3]), num_blocks[3], stride=(2, 1)
        )
        self.conv1x1 = nn.Conv2d(int(384 * width_multiplier[3]), 256, 1, 1)
        self._bn1 = nn.BatchNorm2d(num_features=256, momentum=0.01, eps=1e-3)

    def _make_stage(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for stride in strides:
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
            blocks.append(
                RepVGGBlock(
                    in_channels=self.in_planes,
                    out_channels=planes,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    groups=cur_groups,
                    deploy=self.deploy,
                )
            )
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.Sequential(*blocks)

    def forward(self, x):
        out = self.stage0(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self._bn1(self.conv1x1(out))
        out = out.transpose(-1, -2)
        out = out.flatten(2)
        out = out.permute(-1, 0, 1)

        return out


def create_RepVGG_A0(deploy=False):
    width_multiplier = [0.75, 0.75, 0.75, 2.5]
    num_blocks = [2, 4, 14, 1]

    return RepVGG(
        num_blocks=num_blocks,
        width_multiplier=width_multiplier,
        override_groups_map=None,
        deploy=deploy,
    )
