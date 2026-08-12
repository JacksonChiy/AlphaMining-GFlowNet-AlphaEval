# 输出目录约定

运行生成物应按以下类别组织，实际文件默认不提交Git：

```text
outputs/
├── checkpoints/
├── models/
├── backtests/
├── figures/
├── reports/
└── logs/
```

历史代码仍可能写入`results/`和`checkpoints/`。本轮整理不移动既有结果；新功能优先采用此结构，迁移必须保持配置兼容。
