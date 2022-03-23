
from glob import glob
import pandas as pd


class Accumulator():
    """
    Is a simple tool, not yet perfect, but working to accumulate metrics across multiple folds.
    However paths are static in the meantime. This is to be very much improved in the future! There is already a new
    version in our "pipeline"/making

    """
    def __init__(self):
        super(Accumulator, self).__init__()

    def log_reg_report_mean_vals_with_std(self, model, dataset_name):

        path = f'../results/{dataset_name}/{model}_Folds/'
        resultpath = f'../results/{dataset_name}/'
        csvs = sorted(glob(path + f'{model}_Fold_*.csv'))
        df = pd.concat([pd.read_csv(f, index_col=0) for f in csvs], ignore_index=False)

        s_std = df.std()
        s_mean = df.mean()

        s_std.to_csv(path + f'{model}_Fold_std_dv.csv', float_format='%.5f', index=True)
        df.to_csv(path + f'{model}_Fold_concat.csv', index=True)
        out = s_mean.copy().astype(str)
        for i in range(s_mean.shape[0]):
            out.iloc[i] = "{:.5f}".format(s_mean.values[i]) + u"\u00B1" + "{:.5f}".format(s_std.values[i])
        out.to_csv(resultpath + f'{model}_Fold_mean_with_std.csv', index=True)

    def log_class_report_mean_vals_with_std(self, model, dataset_name):

        path = f'../results/{dataset_name}/{model}_Folds/'
        resultpath = f'../results/{dataset_name}/'
        csvs = sorted(glob(path + f'{model}_Fold_*.csv'))
        df = pd.concat([pd.read_csv(f, index_col=0) for f in csvs], ignore_index=False)

        df_std = df.groupby(level=0).std()
        df_mean = df.groupby(level=0).mean()

        df_std.to_csv(path + f'{model}_Fold_std_dv.csv', float_format='%.5f', index=True)
        df.to_csv(path + f'{model}_Fold_concat.csv', index=True)
        out = df_mean.copy().astype(str)
        for i in range(df_mean.shape[0]):
            for j in range(df_mean.shape[1]):
                out.iloc[i, j] = "{:.5f}".format(df_mean.values[i, j]) + u"\u00B1" + "{:.5f}".format(
                    df_std.values[i, j])
        df_out = pd.DataFrame(out)
        df_out.to_csv(resultpath + f'{model}_Fold_mean_with_std.csv', index=True)

    def log_class_report_mean_vals(self, model, dataset_name):

        path = f'../results/{dataset_name}/{model}_Folds/'
        resultpath = f'../results/{dataset_name}/'
        csvs = sorted(glob(path + f'{model}_Fold_*.csv'))
        df = pd.concat([pd.read_csv(f, index_col=0) for f in csvs], ignore_index=False)
        df.to_csv(path + f'{model}_Fold_concat.csv', index=True)

        df_mean = df.groupby(level=0).mean()
        df_out = pd.DataFrame(df_mean)
        df_out.to_csv(resultpath + f'{model}_Fold_mean.csv', float_format='%.5f', index=True)

    def log_timings_mean_vals(self, model, dataset_name):

        path = f'../results/{dataset_name}/{model}_Folds/'
        resultpath = f'../results/{dataset_name}/'
        csvs = sorted(glob(path + f'Timing_{model}_Fold_*.csv'))
        df = pd.concat([pd.read_csv(f, index_col=0) for f in csvs], ignore_index=False)
        df.to_csv(path + f'Timing_{model}_Fold_concat.csv', index=True)

        # df.timing = pd.to_datetime(df.timing).values.astype(np.int64)
        #
        # mean = df.groupby('model').mean()
        # may deviate at averaging due to overflows, but microseconds is reliable
        mean_timing = df.groupby('model')['timing'].mean()[0]  #pd.to_datetime(.timing)
        df_out = pd.DataFrame({'Average_Timing_Seconds': mean_timing}, index=[1])
        df_out.to_csv(resultpath + f'Timing_{model}_Fold_mean.csv', float_format='%.5f', index=True)


if __name__ == '__main__':
    acc = Accumulator()
    acc.log_class_report_mean_vals_with_std(traditional_models_to_run=[],
                                            DATSETS_TO_RUN=[])
