# Semantic-Dehazing

## ExperimentOne


下面给出一个可以直接粘贴到 GitHub README 的带颜色突出显示的 Markdown 表格版本。该表格使用内嵌的 HTML span 来实现单元格颜色和文本颜色，适用于大多数 GitHub 渲染场景。若你的 readme 渲染器不支持内联 HTML，可以使用后面的纯 Markdown 版本作为备用。

直接可用的 GitHub README 版本（带颜色）
| Loss variant | PSNR (dB) | SSIM | BestStep | 状态 |
|---|---:|---:|---:|---:|
| L1Loss | 19.64 | 0.598 | 43050 | <span style="color:green">已完成</span> |
| SkyL1Loss | <span style="background-color:#d4f7d4;">20.73</span> | 0.587 | <span style="background-color:#d4f7d4;">59400</span> | <span style="color:green">已完成</span> |
| L1+RegionSSIM | 19.83 | <span style="background-color:#d4f7d4;">0.610</span> | 49200 | <span style="color:green">已完成</span> |
| L1+GlobalSSIM | - | - | - | <span style="color:orange">尚未开始</span> |
| SkyL1+RegionSSIM | - | - | - | <span style="color:orange">尚未开始</span> |
| SkyL1+GlobalSSIM | - | - | - | <span style="color:orange">尚未开始</span> |

说明
- 最大 PSNR 的单元格：SkyL1Loss 的 20.73 使用绿色背景突出（可通过背景色 #d4f7d4 查看）。
- 最大 SSIM 的单元格：L1+RegionSSIM 的 0.610 使用绿色背景突出。
- 最大 BestStep 的单元格：SkyL1Loss 的 59400 使用绿色背景突出。
- 已完成使用绿色文本，尚未开始使用橙色文本。

纯 Markdown 备用版本（不含 HTML 的文本颜色）
| Loss variant | PSNR (dB) | SSIM | BestStep | 状态 |
|---|---:|---:|---:|---:|
| L1Loss | 19.64 | 0.598 | 43050 | 已完成 |
| SkyL1Loss | 20.73 | 0.587 | 59400 | 已完成 |
| L1+RegionSSIM | 19.83 | 0.610 | 49200 | 已完成 |
| L1+GlobalSSIM | - | - | - | 尚未开始 |
| SkyL1+RegionSSIM | - | - | - | 尚未开始 |
| SkyL1+GlobalSSIM | - | - | - | 尚未开始 |
