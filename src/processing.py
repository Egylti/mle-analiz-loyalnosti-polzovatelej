import pandas as pd
from scipy.stats import chi2_contingency


class DataProcessor:
    def __init__(self, orders_path, rates_path):
        self.orders_path = orders_path
        self.rates_path = rates_path
        self.orders = None
        self.rates = None
        self.data = None
        self.profiles = None

    def load_data(self):
        self.orders = pd.read_csv(self.orders_path)
        self.rates = pd.read_csv(self.rates_path)

        self.orders['order_dt'] = pd.to_datetime(
            self.orders['order_dt']
        ).dt.normalize()
        self.orders['order_ts'] = pd.to_datetime(self.orders['order_ts'])
        self.rates['rate_date'] = pd.to_datetime(
            self.rates['data']
        ).dt.normalize()

    def preprocess_data(self):
        data = self.orders.copy()
        rates = self.rates[['rate_date', 'nominal', 'curs']].copy()

        data = data.merge(
            rates,
            left_on='order_dt',
            right_on='rate_date',
            how='left'
        )

        data['revenue'] = pd.to_numeric(data['revenue'], errors='coerce')
        data['nominal'] = pd.to_numeric(data['nominal'], errors='coerce')
        data['curs'] = pd.to_numeric(data['curs'], errors='coerce')

        data['revenue_rub'] = data['revenue']
        kzt_orders = data['currency_code'].str.lower() == 'kzt'
        data.loc[kzt_orders, 'revenue_rub'] = (
            data.loc[kzt_orders, 'revenue']
            * data.loc[kzt_orders, 'curs']
            / data.loc[kzt_orders, 'nominal']
        )

        duplicate_fields = [
            'user_id',
            'order_ts',
            'device_type_canonical',
            'service_name',
            'event_id',
            'revenue'
        ]
        duplicates = data.duplicated(subset=duplicate_fields).sum()
        data = data.drop_duplicates(subset=duplicate_fields)
        data = data.dropna(
            subset=['user_id', 'order_id', 'order_dt', 'revenue_rub']
        )

        self.data = data.reset_index(drop=True)

        print(f'Удалено совпадающих заказов: {duplicates}')
        print(f'Заказов после обработки: {len(self.data):,}')

        return self.data

    def create_user_profiles(self):
        data = self.data.sort_values(['user_id', 'order_ts'])

        profiles = data.groupby('user_id', as_index=False).agg(
            total_orders=('order_id', 'nunique'),
            total_revenue_rub=('revenue_rub', 'sum'),
            avg_revenue_rub=('revenue_rub', 'mean'),
            first_order_dt=('order_dt', 'min'),
            first_event_type=('event_type_main', 'first'),
            first_device=('device_type_canonical', 'first'),
            first_region=('region_name', 'first')
        )

        profiles['is_repeat'] = profiles['total_orders'] > 1
        self.profiles = profiles

        return profiles

    def calculate_retention(self):
        profiles = self.create_user_profiles()

        features = {
            'first_event_type': 'Тип первого мероприятия',
            'first_device': 'Устройство первого заказа',
            'first_region': 'Регион первого заказа'
        }

        results = []

        for column, name in features.items():
            table = profiles.groupby(column, dropna=False).agg(
                users=('user_id', 'nunique'),
                return_rate=('is_repeat', 'mean')
            ).reset_index()

            table = table.rename(columns={column: 'segment'})
            table['feature'] = name
            results.append(
                table[['feature', 'segment', 'users', 'return_rate']]
            )

        return pd.concat(results, ignore_index=True)

    def check_hypotheses(self):
        profiles = self.create_user_profiles()
        results = {}

        event_types = (
            profiles['first_event_type']
            .fillna('')
            .str.lower()
        )

        event_group = pd.Series(index=profiles.index, dtype='object')
        event_group[event_types.str.contains('спорт')] = 'спорт'
        event_group[event_types.str.contains('концерт')] = 'концерты'

        event_data = profiles[event_group.notna()].copy()
        event_data['event_group'] = event_group[event_group.notna()]

        event_rates = event_data.groupby('event_group')['is_repeat'].mean()
        event_table = pd.crosstab(
            event_data['event_group'],
            event_data['is_repeat']
        )

        if event_table.shape[0] == 2 and event_table.shape[1] == 2:
            event_p_value = chi2_contingency(event_table)[1]
        else:
            event_p_value = float('nan')

        region_table = pd.crosstab(
            profiles['first_region'],
            profiles['is_repeat']
        )

        if region_table.shape[0] > 1 and region_table.shape[1] == 2:
            region_p_value = chi2_contingency(region_table)[1]
        else:
            region_p_value = float('nan')

        results['event_return_rates'] = event_rates
        results['event_p_value'] = event_p_value
        results['region_p_value'] = region_p_value

        return results

    def correlation_analysis(self):
        profiles = self.create_user_profiles()
        columns = [
            'total_orders',
            'total_revenue_rub',
            'avg_revenue_rub'
        ]

        return profiles[columns].corr(method='spearman')