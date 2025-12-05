def custom_reward_fn(data_source, solution_str, ground_truth, extra_info=None):
    """自定义奖励函数模板
    
    参数:
        data_source: 数据集标识符
        solution_str: 模型生成的解答
        ground_truth: 标准答案
        extra_info: 额外信息字典
        
    返回:
        float: 评分结果
    """
    # 实现自定义评分逻辑
    score = 1
    return score