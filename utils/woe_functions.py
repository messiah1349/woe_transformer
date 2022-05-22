from copy import copy
import pandas as pd
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
from scipy.stats import chisquare
from typing import Tuple, List


def check_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    for avoid pandas warnings
    """
    ix_name = df.index.name
    if ix_name is None:
        ix_name = 'index'
    return df.reset_index().set_index(ix_name)


def _draw_woe(woe_table: pd.DataFrame, ylim: float, show: bool, save: bool, path: str) -> None:
    """draw woe variables visualisation"""
    for col in woe_table['column_name'].unique():
        woe_col = woe_table[woe_table['column_name'] == col].set_index('bucket_intervals')
        woe_col.index = woe_col.index.rename(col)

        fig0, ax0 = plt.subplots()
        ax1 = ax0.twinx()

        woe_col['mean_target'].plot(kind='line', c='r', ax=ax1, ylim=ylim)
        woe_col['total'].plot(kind='bar', stacked=True, ax=ax0)

        if save:
            if not path:
                print('there is no path')
                return

            plt.savefig(path + col, bbox_inches='tight')
            plt.close()
            continue

        if show:
            plt.show()


class WoeTransformer:

    def __init__(self,
                 df: pd.DataFrame,
                 features: List[str],
                 default_flag_df: str,
                 model_type: str = 'classification',
                 borders: list = None,
                 unique_value_cnt: int = 30,
                 min_bins: int = 2,
                 max_bins: int = 5,
                 min_share_of_bucket_size: float = .05,
                 min_sample_in_bucket: int = 100,
                 bad_rate_min_diff: float = .005,
                 check_monotone: bool = True):

        self.default_flag_name = default_flag_df.columns[0]
        self.train_df = default_flag_df.join(df[features], how='inner')
        self.features = features
        self.model_type = model_type
        self.borders = borders

        self.woe_kwargs = {
            'unique_value_cnt': unique_value_cnt,
            'min_bins': min_bins,
            'max_bins': max_bins,
            'min_share_of_bucket_size': min_share_of_bucket_size,
            'min_sample_in_bucket': min_sample_in_bucket,
            'bad_rate_min_diff': bad_rate_min_diff,
            'check_monotone': check_monotone
        }

        self.woe_val, self.woe_table, self.not_woe_features = self._woe_transform()

    def draw_buckets(self,
                     ylim: Tuple[float, float] = (.02, .21),
                     show: bool = True,
                     save: bool = False,
                     path: str = None
                     ):
        _draw_woe(self.woe_table, ylim, show, save, path)

    def _woe_transform(self) -> object:
        """
        run woe transformation
        """

        woe_val = pd.DataFrame()
        woe_table = pd.DataFrame()
        not_woe_features = []

        for column in self.features:
            categorical_flg = True if self.train_df[column].dtype == 'O' else False
            woe_vals_col, woe_table_col = self._get_woe_values(column, categorical_flg)

            if woe_vals_col is None:
                not_woe_features.append(column)
            else:
                woe_val = pd.concat([woe_val, woe_vals_col], axis=1)
                woe_table = pd.concat([woe_table, woe_table_col], axis=0)

        woe_table = self._corr_woe_table(woe_table)

        return woe_val, woe_table, not_woe_features

    @staticmethod
    def _corr_woe_table(woe_table: pd.DataFrame):
        return woe_table[woe_table['total'] > 0]

    def _get_woe_values(self, column_name: str, categorical_flg: bool) -> Tuple:

        if self.borders is None:
            borders_input = {}
        else:
            borders_input = copy(self.borders)

        pre_woe_df = self.train_df[[column_name, self.default_flag_name]]
        pre_woe_df = check_index(pre_woe_df)

        if categorical_flg:
            pre_woe_df['bucket_intervals'] = \
                pre_woe_df[column_name].astype('category').cat.add_categories("nul").fillna("nul")
        else:
            if column_name in borders_input:
                borders = borders_input[column_name]
            else:
                borders = self._calc_best_borders(column_name)

            pre_woe_df['bucket_intervals'] = pd.cut(pre_woe_df[column_name], borders, right=False, duplicates='drop')
            pre_woe_df['bucket_intervals'] = pre_woe_df['bucket_intervals'].cat.add_categories("nul").fillna("nul")

        if self.model_type == 'classification':
            woe_table = self._calc_woe_table(pre_woe_df, column_name)
            woe_dict = woe_table.set_index('bucket_intervals')['WOE'].to_dict()
            woe_values = pre_woe_df['bucket_intervals'].apply(lambda x: woe_dict[x]).astype('float').rename(column_name)
        elif self.model_type == 'regression':
            woe_table = self._calc_woe_table_regression(pre_woe_df, column_name)
            woe_values = pd.DataFrame()

        return woe_values, woe_table

    def _calc_woe_table(self, pre_woe_df: pd.DataFrame, column_name: str) -> pd.DataFrame:
        """
        calc table with all useful statistics for woe breakdown
        :param pre_woe_df:
        :param column_name:
        :param default_column_name:
        :return: table with statistics
        """
        agg = {
            column_name: ['min', 'max'],
            self.default_flag_name: ['sum', 'size']
        }

        pre_woe_gr = pre_woe_df.groupby('bucket_intervals').agg(agg)
        pre_woe_gr.columns = ['minVal', 'maxVal', 'bads', 'total']
        pre_woe_gr['badRate'] = pre_woe_gr['bads'] / pre_woe_gr['bads'].sum()
        pre_woe_gr['goods'] = pre_woe_gr['total'] - pre_woe_gr['bads']
        pre_woe_gr['goodRate'] = pre_woe_gr['goods'] / pre_woe_gr['goods'].sum()
        pre_woe_gr['mean_target'] = pre_woe_gr['bads'] / pre_woe_gr['total']

        rate_ratio = pre_woe_gr['badRate'] / pre_woe_gr['goodRate']
        rate_ratio = np.where(rate_ratio > .001, rate_ratio, .001)
        pre_woe_gr['WOE'] = np.log(rate_ratio) * 100
        pre_woe_gr['WOE'] = pre_woe_gr['WOE'].replace(float('inf'), 0).replace(-float('inf'), 0)

        pre_woe_gr = pre_woe_gr.reset_index()
        pre_woe_gr['column_name'] = column_name

        return pre_woe_gr

    def _calc_woe_table_regression(self, pre_woe_df: pd.DataFrame, column_name: str) -> pd.DataFrame:
        """
        calc table with all useful statistics for woe breakdown, use regression method
        :param pre_woe_df:
        :param column_name:
        :param default_column_name:
        :return: table with statistics
        """
        agg = {
            column_name: ['min', 'max'],
            self.default_flag_name: ['mean', 'size']
        }

        pre_woe_gr = pre_woe_df.groupby('bucket_intervals').agg(agg)
        pre_woe_gr.columns = ['minVal', 'maxVal', 'mean_target', 'total']

        pre_woe_gr = pre_woe_gr.reset_index()
        pre_woe_gr['column_name'] = column_name

        return pre_woe_gr

    def _calc_best_borders(self, cut_column_name: str) -> np.ndarray:
        """
        for cut column find best bucket breakdown and calculate borders for cutting column
        :param cut_column_name: name of feature column in dataframe
        :return: values of borders of cut column for future bucket breakdown
        """
        unique_value_cnt = self.woe_kwargs['unique_value_cnt']

        clean_df = \
            self.train_df[self.train_df[cut_column_name].notnull()][[cut_column_name, self.default_flag_name]]\
            .sort_values(cut_column_name)
        clean_array = clean_df.values

        column_array = clean_array[:, 0]  # same as clean_df[column_name].values

        step = max(int(column_array.size / unique_value_cnt), 1)
        # print(step, column_array.size)
        quantile_indexes = np.arange(step, column_array.size, step)
        quantiles = np.sort(column_array).take(quantile_indexes)

        # drop duplicates by quantiles
        gr_df = pd.DataFrame({'quantile_indexes': quantile_indexes, 'quantiles': quantiles})
        quantile_indexes = gr_df.groupby('quantiles').head(1)['quantile_indexes'].values

        best_comb, chis = self._find_best_combination(clean_array, quantile_indexes)

        if not best_comb:
            return self._add_min_max_to_borders(np.array([]), column_array.min(), column_array.max())

        border_ixs = quantile_indexes.take(best_comb)
        borders = column_array.take(border_ixs)
        borders = self._add_min_max_to_borders(borders, column_array.min(), column_array.max())

        return borders

    def _get_chi_square_by_buckets(self, splitted_buckets: list):
        bucket_info = np.array([(bucket[:, 1].sum(), bucket.shape[0]) for bucket in splitted_buckets])

        bucket_sizes = bucket_info[:, 1]
        cnt_of_ones = bucket_info[:, 0]
        cnt_of_zeros = bucket_sizes - cnt_of_ones

        chi_square = None

        if self._check_buckets(cnt_of_zeros, cnt_of_ones, bucket_sizes):
            chi_square = self._calc_chi_square_hand(cnt_of_zeros, cnt_of_ones)

        return chi_square

    def _get_mse_by_buckets(self, splitted_buckets: list, average_target: float):
        """
        calculate mse metric for bucket chose distribution
        :param splitted_buckets: information about variable splitting
        :param average_target: observed average target
        :return: mse value
        """
        bucket_info = np.array([(
                np.square(bucket[:, 1] - average_target).mean(),
                bucket[:, 1].mean(),
                bucket.shape[0]
            ) for bucket in splitted_buckets])

        mses = bucket_info[:, 0]
        averages = bucket_info[:, 1]
        bucket_sizes = bucket_info[:, 2]

        mse = None

        if self._check_buckets_regression(averages, bucket_sizes):
            bucket_frequencies = bucket_sizes / bucket_sizes.sum()
            mse = (mses * bucket_frequencies).sum()

        return mse

    def _find_best_combination(self, clean_array: np.ndarray, quantile_indexes: np.ndarray) -> \
            Tuple[np.ndarray, float]:
        """
        find best combination of breakdown at buckets by brute force indexes combinations and calculating chi square
        statistic for this breakdown
        :param clean_array: values of initial columns
        :param quantile_indexes: allowed indexes for brute force
        :return: indexes for better combinations, value of chi square for this best combination
        """
        best_stat_value = 0
        best_comb = None

        quantiles_size = quantile_indexes.size

        min_bins = self.woe_kwargs['min_bins']
        max_bins = self.woe_kwargs['max_bins']

        for num_of_borders in range(min_bins - 1, max_bins):
            for comb in combinations(range(quantiles_size), num_of_borders):

                # find indexes of borders
                current_interval_border_ixs = quantile_indexes.take(comb)

                # split dataframe by borders. Analogue of groupby by buckets with this borders
                splitted_buckets = np.split(clean_array, current_interval_border_ixs, axis=0)

                # get array with two columns, num of rows == num of buckets.
                # 0s columns - sum_of_1, 1s column - size of bucket

                if self.model_type == 'classification':
                    current_statistic = self._get_chi_square_by_buckets(splitted_buckets)
                elif self.model_type == 'regression':
                    average_target = clean_array[:, 0].mean()
                    current_statistic = self._get_mse_by_buckets(splitted_buckets, average_target)

                if current_statistic and current_statistic > best_stat_value:
                    best_stat_value = current_statistic
                    best_comb = comb

        return best_comb, best_stat_value

    @staticmethod
    def _calc_chi_square(cnt_of_zeros: np.ndarray, cnt_of_ones: np.ndarray) -> float:
        """
        calc chiSquare test for observed and expected count of ones and zeros at buckets
        than value more than better buckets approximate initial distribution
        :param cnt_of_zeros:
        :param cnt_of_ones:
        :return: value of chiSquare
        """
        observed_values = np.concatenate([cnt_of_zeros, cnt_of_ones])

        cnt_of_all_zeros = cnt_of_zeros.sum()
        cnt_of_all_ones = cnt_of_ones.sum()
        cnt_of_elements = cnt_of_all_zeros + cnt_of_all_ones
        cnt_of_zeros_expected = cnt_of_zeros * cnt_of_all_zeros / cnt_of_elements
        cnt_of_ones_expected = cnt_of_ones * cnt_of_all_ones / cnt_of_elements
        expected_values = np.concatenate([cnt_of_zeros_expected, cnt_of_ones_expected])

        chs = chisquare(observed_values, f_exp=expected_values)[0]

        return chs

    @staticmethod
    def _calc_chi_square_hand(cnt_of_zeros: np.ndarray, cnt_of_ones: np.ndarray) -> float:
        observed = np.matrix(np.stack([cnt_of_zeros, cnt_of_ones]).T)

        x = observed.sum(0)
        y = observed.sum(1)
        expected = y * x / observed.sum()

        chi_squared = (np.square(observed - expected) / expected).sum()
        return chi_squared

    def _check_buckets(self, cnt_of_zeros: np.ndarray, cnt_of_ones: np.ndarray, bucket_sizes: np.ndarray) -> bool:
        """
        checking row of conditions. if at least one incorrect then return False
        params - count of zeros, ones and total size of buckets - numpy arrays of the same size (count of buckets)
        :param cnt_of_zeros:
        :param cnt_of_ones:
        :param bucket_sizes:
        :param kwargs: see WoeTransformer initial parameters
        :return: condition that all rules correct
        """
        # there is no zeros
        if min(cnt_of_zeros.min(), cnt_of_ones.min()) < .1:
            return False

        # limit share of size of least bucket
        min_share_of_bucket_size = self.woe_kwargs['min_share_of_bucket_size']

        if (bucket_sizes / bucket_sizes.sum()).min() < min_share_of_bucket_size:
            return False

        # limit size of least bucket
        min_sample_in_bucket = self.woe_kwargs['min_sample_in_bucket']

        if bucket_sizes.min() < min_sample_in_bucket:
            return False

        # bad rate difference between two neighbour buckets should be more then value of parameter
        bad_rate_min_diff = self.woe_kwargs['bad_rate_min_diff']

        bad_rates = cnt_of_ones / bucket_sizes
        bad_rate_bucket_diffs = bad_rates[:-1] - bad_rates[1:]

        if np.abs(bad_rate_bucket_diffs).min() < bad_rate_min_diff:
            return False

        # bad rate values should be monotone
        check_monotone = self.woe_kwargs['check_monotone']

        if check_monotone:

            digit_set = {x > 0 for x in bad_rate_bucket_diffs}
            if len(digit_set) > 1:
                return False

        return True

    def _check_buckets_regression(self, averages: np.ndarray, bucket_sizes: np.ndarray) -> bool:

        # limit share of size of least bucket
        min_share_of_bucket_size = self.woe_kwargs['min_share_of_bucket_size']

        if (bucket_sizes / bucket_sizes.sum()).min() < min_share_of_bucket_size:
            return False

        # limit size of least bucket
        min_sample_in_bucket = self.woe_kwargs['min_sample_in_bucket']

        if bucket_sizes.min() < min_sample_in_bucket:
            return False

        # average target difference between two neighbour buckets should be more then value of parameter
        bad_rate_min_diff = self.woe_kwargs['bad_rate_min_diff']

        bad_rate_bucket_diffs = averages[:-1] - averages[1:]

        if np.abs(bad_rate_bucket_diffs).min() < bad_rate_min_diff:
            return False

        # bad rate values should be monotone
        check_monotone = self.woe_kwargs['check_monotone']

        if check_monotone:

            digit_set = {x > 0 for x in bad_rate_bucket_diffs}
            if len(digit_set) > 1:
                return False

        return True

    @staticmethod
    def _add_min_max_to_borders(borders: np.ndarray, min_value: float = None, max_value: float = None,\
                               precision: float = .001) -> np.ndarray:
        """
        concat array with minimum and maximum values of borders
        :param borders: inner borders
        :param min_value:
        :param max_value:
        :param precision:
        :return: concatenated array [min_value, *borders, max_value]
        """
        if min_value is not None:
            borders = np.append([min_value - precision], borders)

        if max_value is not None:
            borders = np.append(borders, [max_value + precision])

        return borders





