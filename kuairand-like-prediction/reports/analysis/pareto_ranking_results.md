# Pareto Reranking Results

Mean NDCG@k across users

## Strategy: click_only
| Target | k=5 | k=10 | k=20 |
|---|---:|---:|---:|
| is_click | 0.9915 | 0.9914 | 0.9914 |
| is_like | 0.0351 | 0.0384 | 0.0390 |
| long_view | 0.6636 | 0.6648 | 0.6651 |

## Strategy: like_only
| Target | k=5 | k=10 | k=20 |
|---|---:|---:|---:|
| is_click | 0.9855 | 0.9873 | 0.9875 |
| is_like | 0.0383 | 0.0400 | 0.0403 |
| long_view | 0.4742 | 0.5087 | 0.5157 |

## Strategy: longview_only
| Target | k=5 | k=10 | k=20 |
|---|---:|---:|---:|
| is_click | 0.9910 | 0.9909 | 0.9909 |
| is_like | 0.0349 | 0.0382 | 0.0388 |
| long_view | 0.6781 | 0.6787 | 0.6788 |

## Strategy: weighted_scalar
| Target | k=5 | k=10 | k=20 |
|---|---:|---:|---:|
| is_click | 0.9901 | 0.9905 | 0.9906 |
| is_like | 0.0338 | 0.0372 | 0.0372 |
| long_view | 0.6737 | 0.6753 | 0.6754 |

## Strategy: pareto_frontier
| Target | k=5 | k=10 | k=20 |
|---|---:|---:|---:|
| is_click | 0.9889 | 0.9898 | 0.9900 |
| is_like | 0.0345 | 0.0378 | 0.0378 |
| long_view | 0.6642 | 0.6689 | 0.6705 |

## Strategy: pareto_weighted
| Target | k=5 | k=10 | k=20 |
|---|---:|---:|---:|
| is_click | 0.9891 | 0.9901 | 0.9902 |
| is_like | 0.0340 | 0.0375 | 0.0375 |
| long_view | 0.6679 | 0.6723 | 0.6736 |
