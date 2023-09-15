from copy import deepcopy
from typing import Tuple, Any, List
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
import lightgbm
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportion_confint

from .woe_functions import WoeTransformer, check_index


def merge_default_flag_with_features(default_flg_df: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
    return default_flg_df.join(features_df, how='inner')


def splitting(data: pd.DataFrame, default_flg: pd.DataFrame, test_size=.3, rs=18):
    data_mer = merge_default_flag_with_features(default_flg, data)

    default_flag_name = default_flg.columns[0]
    y = data_mer[default_flag_name]
    X = data_mer.drop(default_flag_name, axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=rs)

    feature_names = list(X_train.columns)

    return feature_names, X_train, y_train, X_test, y_test


def categorical_column_to_label(train: pd.DataFrame, test: pd.DataFrame) -> \
        Tuple[pd.DataFrame, pd.DataFrame, dict]:
    data = pd.concat([train, test])
    train_ret = train.copy()
    test_ret = test.copy()

    dt = data.dtypes
    obj_cols = dt[dt == 'object'].index.tolist()

    le_dict = {}

    for col in obj_cols:
        le = LabelEncoder()
        le.fit(data[col])
        train_ret[col] = le.transform(train_ret[col])
        test_ret[col] = le.transform(test_ret[col])
        le_dict[col] = le

    return train_ret, test_ret, le_dict


def get_corr_columns(df: pd.DataFrame, corr_border: float = .5) -> pd.DataFrame:
    corr_df = df.corr().unstack().apply(abs).reset_index()
    corr_df.columns = ['col_1', 'col_2', 'correlation']
    corr_df = corr_df[corr_df['col_1'] > corr_df['col_2']]
    corr_df = corr_df[corr_df['correlation'] >= corr_border]
    corr_df = corr_df.sort_values('correlation', ascending=False)

    return corr_df


def get_feature_auc_diff(lr: LogisticRegression):
    features = lr.feature_names
    aucs_reverses = {}

    for feat in features:
        lr_current = deepcopy(lr)
        lr_current.feature_names = [col for col in features if col != feat]
        lr_current.fit()
        aucs_reverses[feat] = lr_current.get_auc()

    res_df = pd.DataFrame(aucs_reverses).T.reset_index()
    res_df.columns = ['column', 'train_auc', 'test_auc']
    res_df = res_df.sort_values('test_auc', ascending=False)

    return res_df


def get_proportion_confint_from_df(row):
    return proportion_confint(row['bad_counts'], row['bucket_size'], .05)


def get_score_bucket_table(score_table: pd.DataFrame, num_buck: int, score_column_name='pr') -> pd.DataFrame:
    score_gr = score_table.reset_index(drop=True)

    agg = {
        'y': ['mean', 'size', 'sum'],
        score_column_name: ['min', 'max']
    }

    score_gr['bucket'] = -1
    score_gr['bucket'] = score_gr.index * num_buck // len(score_gr)
    score_gr = score_gr.groupby('bucket').agg(agg)
    score_gr.columns = ['bad_rate_mean', 'bucket_size', 'bad_counts', 'min_pr', 'max_pr']
    score_gr['low_confidence_border'] = score_gr.apply(get_proportion_confint_from_df, axis=1).str[0]
    score_gr['upp_confidence_border'] = score_gr.apply(get_proportion_confint_from_df, axis=1).str[1]

    return score_gr


def check_interval_entry(value: Any, interval: Any) -> bool:

    # categorical case
    if isinstance(value, str):
        if value == interval:
            return True
        else:
            return False

    # number case
    if interval == 'nul' and (np.isnan(value) or value is None):
        return True
    if interval != 'nul' and not np.isnan(value) and value in interval:
        return True
    else:
        return False


def inverse_le(feature: pd.Series, le_vals: dict):
    return feature.apply(lambda x: le_vals[x])


def roc_curve(preds: np.ndarray, y: np.ndarray) -> None:
    fpr, tpr, threshold = metrics.roc_curve(y, preds)
    roc_auc = metrics.auc(fpr, tpr)

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


class Model:

    def __init__(self,
                 feature_names: List[str],
                 X_train: pd.DataFrame,
                 y_train: pd.Series,
                 X_test: pd.DataFrame,
                 y_test: pd.Series
                 ):
        self.X_train = X_train[feature_names]
        self.y_train = y_train
        self.X_test = X_test[feature_names]
        self.y_test = y_test
        self.feature_names = feature_names
        self.auc_train = None
        self.auc_test = None
        self.score_table = None

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def _calc_auc(self):
        self.auc_train = metrics.roc_auc_score(self.y_train, self.pr_train)
        self.auc_test = metrics.roc_auc_score(self.y_test, self.pr_test)

    def get_auc(self):
        self._calc_auc()
        return self.auc_train, self.auc_test

    def print_auc(self):
        self._calc_auc()
        print(f"train_auc = {self.auc_train}, \
                test_auc = {self.auc_test}")

    def draw_roc_train(self):
        roc_curve(self.pr_train, self.y_train)

    def draw_roc_test(self):
        roc_curve(self.pr_test, self.y_test)

    def draw_roc(self):
        roc_curve(self.pr_train, self.y_train)
        roc_curve(self.pr_test, self.y_test)

    def calc_score_table(self):

        score_table_test = pd.DataFrame({'y': self.y_test, 'pr': self.pr_test}).set_index(
            self.y_test.index)
        score_table_test['test_flg'] = 1
        score_table_train = pd.DataFrame({'y': self.y_train, 'pr': self.pr_train}).set_index(
            self.y_train.index)
        score_table_train['test_flg'] = 0
        self.score_table = pd.concat([score_table_train,score_table_test], axis=0).sort_values('pr', ascending=False)

    def calc_score_bucket_train(self, num_buck=5):
        train_scores = self.score_table[self.score_table['test_flg']==0]
        return get_score_bucket_table(train_scores, num_buck)

    def calc_score_bucket_test(self, num_buck=5):
        test_scores = self.score_table[self.score_table['test_flg'] == 1]
        return get_score_bucket_table(test_scores, num_buck)

    def calc_score_bucket_all(self, num_bucket=5):
        return get_score_bucket_table(self.score_table, num_bucket)


class SimpleLGM(Model):

    def __init__(self,
                 feature_names: List[str],
                 X_train: pd.DataFrame,
                 y_train: pd.Series,
                 X_test: pd.DataFrame,
                 y_test: pd.Series
                 ):
        super().__init__(feature_names, X_train, y_train, X_test, y_test)

        self.X_train_cat, self.x_test_cat, self.le_dict = categorical_column_to_label(X_train, X_test)

        self.lgtrain = lightgbm.Dataset(self.X_train_cat, label=self.y_train)
        self.lgtest = lightgbm.Dataset(self.x_test_cat, label=self.y_test)

    def _calc_preds(self):
        self.pr_train = self.lgb.predict(self.X_train_cat)
        self.pr_test = self.lgb.predict(self.x_test_cat)

    def fit_lgb(self, params):
        evals_result = {}
        model = lightgbm.train(params, self.lgtrain, 10000, valid_sets=[self.lgtest])

        self.lgb = model
        self._calc_preds()

    def get_fi(self):
        fi = pd.DataFrame({'fi': self.lgb.feature_importance(), 'col': self.feature_names}) \
            .sort_values('fi', ascending=False)
        return fi

#    def draw_shap(self):
#        background_adult = shap.maskers.Independent(self.X_train_cat, max_samples=100)
#        explainer = shap.Explainer(self.lgb, background_adult)
#        shap_values = explainer(self.X_train_cat)
#
#        # set a display version of the data to use for plotting (has string values)
#        shap_values.display_data = shap.datasets.adult(display=True)[0].values
#
#        shap.plots.beeswarm(shap_values, max_display=14)


class LogReg(Model):

    def __init__(self,
                 feature_names: List[str],
                 X_train: pd.DataFrame,
                 y_train: pd.Series,
                 X_test: pd.DataFrame,
                 y_test: pd.Series,
                 scaling: bool = True,
                 **lr_parameters
                 ):

        super().__init__(feature_names, X_train, y_train, X_test, y_test)

        self.scaling = scaling
        self.lr_parameters = lr_parameters
        self.lr = LogisticRegression(**self.lr_parameters)

    def _calc_preds(self):
        if self.scaling:
            self.pr_train = self.lr.predict_proba(self.X_train_ss)[:, 1]
            self.pr_test = self.lr.predict_proba(self.X_test_ss)[:, 1]
        else:
            self.pr_train = self.lr.predict_proba(self.X_train[self.feature_names])[:, 1]
            self.pr_test = self.lr.predict_proba(self.X_test[self.feature_names])[:, 1]

    def fit(self):

        if self.scaling:
            self.ss = StandardScaler()
            self.ss.fit(self.X_train[self.feature_names])

            self.X_train_ss = self.ss.transform(self.X_train[self.feature_names])
            self.X_test_ss = self.ss.transform(self.X_test[self.feature_names])

            self.lr.fit(self.X_train_ss, self.y_train)

        else:
            self.lr.fit(self.X_train[self.feature_names], self.y_train)

        self._calc_preds()
        self.calc_score_table()

    def get_woe_output(self,
                       woe_table: pd.DataFrame,
                       src_data: pd.DataFrame,
                       factor_koeff: int = 20,
                       offset_start: int = 600
                       ):

        if factor_koeff is None and offset_start is None:
            factor = -1
            intercept = self.lr.intercept_[0]

        else:
            factor = factor_koeff / np.log(2)
            offset = offset_start - (factor * np.log(50))
            intercept = offset - self.lr.intercept_[0] * factor

        coeff_dict = dict(zip(self.feature_names, self.lr.coef_[0]))
        woe_filtered = woe_table[woe_table['column_name'].isin(self.feature_names)]
        woe_filtered = check_index(woe_filtered)
        woe_filtered['coeff'] = woe_filtered['column_name'].apply(lambda x: coeff_dict[x])
        woe_filtered['score_value'] = - woe_filtered['WOE'] * woe_filtered['coeff'] * factor
        
        ix_name = src_data.index.name
        
        src_melt = src_data.reset_index().melt(value_vars=self.feature_names, var_name='column_name', id_vars=ix_name)
        src_melt = src_melt.merge(woe_filtered[['column_name', 'bucket_intervals', 'score_value']])
        src_melt['interval_entry'] = src_melt.apply(
            lambda row: check_interval_entry(row['value'], row['bucket_intervals']), axis=1)

        fin_score = \
            src_melt[src_melt['interval_entry']].pivot_table(index='order_key', values='score_value',
                                                                 columns='column_name')
        fin_score.columns = [col + '_point' for col in fin_score.columns]
        fin_score['intercept'] = intercept
        fin_score['score_value'] = fin_score.sum(axis=1)

        return woe_filtered, fin_score


class ModelPipeline(Model):

    def __init__(self,
                 feature_names: List[str],
                 X_train: pd.DataFrame,
                 y_train: pd.Series,
                 X_test: pd.DataFrame,
                 y_test: pd.Series,
                 lgb_params: dict,
                 fi_border: int = 2,
                 woe_columns: List[str] = None,
                 draw_woe_pictures: bool = False,
                 woe_user_borders: dict = None,
                 **woe_kwargs
                 ):

        super().__init__(feature_names, X_train, y_train, X_test, y_test)
        self.lgb_all_features = SimpleLGM(feature_names, X_train, y_train, X_test, y_test)
        self.lgb_params = lgb_params
        self.fi_border = fi_border
        self.woe_columns = woe_columns
        self.woe_kwargs = woe_kwargs
        self.draw_woe_pictures = draw_woe_pictures
        self.woe_user_borders = woe_user_borders

    def build_lgb_all_features(self):

        self.lgb_all_features.fit_lgb(params=self.lgb_params)
        self.lgb_all_features.draw_roc_train()
        self.lgb_all_features.draw_roc_test()

    def get_top_features_lgb(self):

        fi_df = self.lgb_all_features.get_fi()
        self.top_lgb_features = fi_df.loc[fi_df['fi'] >= self.fi_border, 'col'].tolist()

    def build_lgb_top_features(self):

        self.lgb_top_features = SimpleLGM(self.top_lgb_features,
                                            self.X_train,
                                            self.y_train,
                                            self.X_test,
                                            self.y_test)
        self.lgb_top_features.fit_lgb(params=self.lgb_params)
        self.lgb_top_features.draw_roc_train()
        self.lgb_top_features.draw_roc_test()

    def get_woe_transform(self):

        if self.woe_columns is None:
            self.woe_columns = self.top_lgb_features

        self.X = pd.concat([self.X_train, self.X_test], axis=0)
        self.y = pd.concat([self.y_train, self.y_test], axis=0).to_frame()

        self.woe = WoeTransformer(self.X, self.woe_columns, self.y, 'classification',
                                  self.woe_user_borders, **self.woe_kwargs)

    def draw_woe(self, y_lim=(.02, .21)):

        self.woe.draw_buckets(y_lim)

    def save_woe_pictures(self, picture_path: str):
        self.woe.draw_buckets(path=picture_path, save=True, show=True)

    def build_lr_for_woe_features(self):

        self.lr_woe = LogReg(*splitting(self.woe.woe_val, self.y), scaling=False)
        self.lr_woe.fit()
        self.lr_woe.draw_roc_train()
        self.lr_woe.draw_roc_test()

    def build(self):

        print('building Lightgbm with all features:')
        self.build_lgb_all_features()
        self.get_top_features_lgb()
        print('top features:')
        for col in self.top_lgb_features:
            print(col)
        print('\n\nbuilding Lightgbm with only top features:')
        self.build_lgb_top_features()

        print('build woe transform:')
        self.get_woe_transform()

        if self.draw_woe_pictures:
            self.draw_woe()

        print('build logistic regression with woe columns')
        self.build_lr_for_woe_features()
