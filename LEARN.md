## 使用装饰器注册类

```py
REWARD_MANAGER_REGISTRY = {}


def register(name):
    """Decorator to register a reward manager class with a given name.

    Args:
        name: `(str)`
            The name of the reward manager.
    """

    def decorator(cls):
        if name in REWARD_MANAGER_REGISTRY and REWARD_MANAGER_REGISTRY[name] != cls:
            raise ValueError(f"Reward manager {name} has already been registered: {REWARD_MANAGER_REGISTRY[name]} vs {cls}")
        REWARD_MANAGER_REGISTRY[name] = cls
        return cls

    return decorator


@register("naive")
class NaiveRewardManager:
    ...
```


