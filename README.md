# skill250-portfolio-rank-strategy
A quantitative trading skill to calculate portfolio rank via sector covariance matrix, eliminate pseudo-diversification.
# skill250 portfolio rank & position control
import numpy as np

def calc_portfolio_rank(sector_cov_matrix, threshold=1e-4):
    """
    Calculate effective portfolio rank
    :param sector_cov_matrix: covariance matrix of sector daily returns
    :param threshold: eigenvalue threshold
    :return: effective rank
    """
    eigen_values, _ = np.linalg.eig(sector_cov_matrix)
    valid_eigen = eigen_values[eigen_values > threshold]
    effective_rank = len(valid_eigen)
    return effective_rank


def filter_high_correlation_sectors(sector_return_data, rank_min=3):
    """
    Iteratively remove highly correlated sectors until portfolio rank >= rank_min
    """
    cov_mat = np.cov(sector_return_data)
    current_rank = calc_portfolio_rank(cov_mat)
    sector_idx_list = list(range(sector_return_data.shape[0]))

    while current_rank < rank_min and len(sector_idx_list) > rank_min:
        cov_mat = np.cov(sector_return_data[sector_idx_list])
        corr_matrix = np.corrcoef(sector_return_data[sector_idx_list])
        np.fill_diagonal(corr_matrix, 0)
        worst_idx = np.argmax(np.max(np.abs(corr_matrix), axis=1))
        sector_idx_list.pop(worst_idx)
        cov_mat = np.cov(sector_return_data[sector_idx_list])
        current_rank = calc_portfolio_rank(cov_mat)
    return sector_idx_list, current_rank


def get_max_single_position(total_capital, ratio=0.25):
    """Single stock position upper limit: 25%"""
    return total_capital * ratio


def trigger_take_profit(current_return, profit_threshold=0.15):
    return current_return >= profit_threshold


def trigger_stop_loss(current_return, loss_threshold=-0.08):
    return current_return <= loss_threshold
