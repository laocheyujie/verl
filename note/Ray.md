# Ray

概念：

- 资源池 `RayResourcePool`：通过不同的资源池名称来划分 CPU、GPU 资源，可以共享资源，比如 `resource_pool_1` 使用 4 个 GPU；`resource_pool_2` 使用 4 个 GPU；`resource_pool_merge` 使用上面两个合并后的结果
- `Worker`：继承了 `Worker` 的类，实际管理 RL 的数据流
- `RayClassWithInitArgs`：把 `Worker` 和初始化参数绑定好的类，用于**延迟初始化 Worker**
- `RayWorkerGroup`：把 `RayResourcePool` 和 `RayClassWithInitArgs` 绑定到一个 Worker 组里，只用于分布式系统的资源调度

流程：

1. 先定义一个 `Worker` 类（为避免每次都用 `execute_all_sync` 下发任务，可以在类方法中加上 `@register(Dispatch.ONE_TO_ALL)` 装饰器）
2. 再结合初始化参数绑定到 `RayClassWithInitArgs`
3. 定义好资源池 `RayResourcePool`
4. 根据资源池 `RayResourcePool` 和 `RayClassWithInitArgs` 创建 Worker 组 `RayWorkerGroup`

具体用法：`examples/ray/tutorial.ipynb`
