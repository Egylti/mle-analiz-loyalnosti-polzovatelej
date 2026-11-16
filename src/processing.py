import pandas as pd
import phik
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

    def preprocess_data(self):
        data = self.orders.copy()

        category_columns = [
            'device_type_canonical', 'currency_code', 'event_name',
            'service_name', 'event_type_main', 'region_name', 'city_name'
        ]

        for column in category_columns:
            if column in data.columns:
                data[column] = data[column].astype('string').str.strip()
                data[column] = data[column].replace('', pd.NA)

        data['order_dt'] = pd.to_datetime(data['order_dt'], errors='coerce')
        data['order_ts'] = pd.to_datetime(data['order_ts'], errors='coerce')
        data['revenue'] = pd.to_numeric(data['revenue'], errors='coerce')

        rates = self.rates.copy()
        rates['rate_date'] = pd.to_datetime(
            rates['data'], errors='coerce'
        ).dt.normalize()
        rates['nominal'] = pd.to_numeric(rates['nominal'], errors='coerce')
        rates['curs'] = pd.to_numeric(rates['curs'], errors='coerce')

        data['rate_date'] = data['order_dt'].dt.normalize()
        data = data.merge(
            rates[['rate_date', 'nominal', 'curs']],
            on='rate_date',
            how='left'
        )

        data['revenue_rub'] = data['revenue']
        is_kzt = data['currency_code'].str.lower().eq('kzt')
        data.loc[is_kzt, 'revenue_rub'] = (
            data.loc[is_kzt, 'revenue']
            * data.loc[is_kzt, 'curs']
            / data.loc[is_kzt, 'nominal']
        )

        duplicate_fields = [
            'user_id', 'order_ts', 'device_type_canonical',
            'service_name', 'event_id', 'revenue_rub'
        ]
        duplicate_mask = data.duplicated(
            subset=duplicate_fields,
            keep='first'
        )
        duplicate_count = int(duplicate_mask.sum())
        data = data.loc[~duplicate_mask].copy()

        missing_revenue_count = int(data['revenue_rub'].isna().sum())
        negative_count = int(data['revenue_rub'].lt(0).sum())
        zero_count = int(data['revenue_rub'].eq(0).sum())

        positive = data.loc[
            data['revenue_rub'].notna() & data['revenue_rub'].gt(0)
        ].copy()

        revenue_q99 = positive['revenue_rub'].quantile(0.99)
        above_revenue_q99 = positive['revenue_rub'].gt(revenue_q99)
        revenue_outliers = int(above_revenue_q99.sum())
        data = positive.loc[~above_revenue_q99].copy()

        data = data.sort_values(
            ['user_id', 'order_ts', 'order_id']
        ).copy()
        data['days_since_prev'] = (
            data.groupby('user_id')['order_dt']
            .diff()
            .dt.days
            .astype('Int64')
        )

        self.data = data.reset_index(drop=True)

        print(f'Совпадающих заказов удалено: {duplicate_count:,}')
        print(f'Отрицательных заказов удалено: {negative_count:,}')
        print(f'Заказов с нулевой выручкой удалено: {zero_count:,}')
        print(f'Строк без выручки удалено: {missing_revenue_count:,}')
        print(f'Заказов выше p99 удалено: {revenue_outliers:,}')
        print(f'Заказов после очистки: {len(self.data):,}')

        return self.data

    def create_user_profiles(self):
        if self.profiles is not None:
            return self.profiles

        data = self.data.sort_values(
            ['user_id', 'order_ts', 'order_id']
        ).copy()
        first_orders = data.groupby('user_id', as_index=False).head(1)

        profiles = data.groupby('user_id').agg(
            first_order_dt=('order_dt', 'min'),
            last_order_dt=('order_dt', 'max'),
            total_orders=('order_id', 'nunique'),
            avg_revenue_rub=('revenue_rub', 'mean'),
            avg_days_between_orders=('days_since_prev', 'mean')
        ).reset_index()

        first_profile = first_orders[
            [
                'user_id', 'device_type_canonical', 'region_name',
                'event_type_main', 'city_name', 'event_name', 'order_ts'
            ]
        ].rename(columns={
            'device_type_canonical': 'first_device',
            'region_name': 'first_region',
            'event_type_main': 'first_event_type',
            'city_name': 'first_city',
            'event_name': 'first_event_name',
            'order_ts': 'first_order_ts'
        })

        profiles = profiles.merge(
            first_profile,
            on='user_id',
            how='left',
            validate='one_to_one'
        )

        profiles['first_weekday'] = profiles['first_order_dt'].dt.dayofweek
        weekday_names = [
            'Понедельник', 'Вторник', 'Среда', 'Четверг',
            'Пятница', 'Суббота', 'Воскресенье'
        ]
        profiles['first_weekday_name'] = profiles['first_weekday'].map(
            dict(enumerate(weekday_names))
        )
        profiles['is_two'] = profiles['total_orders'].ge(2).astype('int8')

        order_count_q99 = profiles['total_orders'].quantile(0.99)
        high_frequency = profiles['total_orders'] > order_count_q99
        removed_users = int(high_frequency.sum())

        excluded_ids = profiles.loc[high_frequency, 'user_id']
        self.data = self.data.loc[
            ~self.data['user_id'].isin(excluded_ids)
        ].copy()
        profiles = profiles.loc[~high_frequency].copy()

        self.profiles = profiles.reset_index(drop=True)

        print(f'p99 числа заказов на пользователя: {order_count_q99:.0f}')
        print(f'Профилей выше p99 удалено: {removed_users:,}')
        print(f'Пользователей после фильтрации: {len(self.profiles):,}')

        return self.profiles

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
                return_rate=('is_two', 'mean')
            ).reset_index()

            table = table.rename(columns={column: 'segment'})
            table['feature'] = name
            results.append(
                table[['feature', 'segment', 'users', 'return_rate']]
            )

        return pd.concat(results, ignore_index=True)

    def check_hypotheses(self):
        profiles = self.create_user_profiles()

        event_labels = (
            profiles['first_event_type']
            .astype('string')
            .str.strip()
            .str.lower()
        )
        event_data = profiles.loc[
            event_labels.isin(['спорт', 'концерты'])
        ].copy()
        event_data['event_group'] = event_labels.loc[event_data.index]

        event_rates = event_data.groupby('event_group')['is_two'].mean()
        event_table = pd.crosstab(
            event_data['event_group'],
            event_data['is_two']
        )
        event_p_value = (
            chi2_contingency(event_table)[1]
            if event_table.shape == (2, 2)
            else float('nan')
        )

        regions = profiles.loc[profiles['first_region'].notna()].copy()
        top_regions = regions['first_region'].value_counts().head(10).index
        regions['region_group'] = regions['first_region'].isin(
            top_regions
        ).map({
            True: 'Топ-10 регионов',
            False: 'Остальные регионы'
        })

        region_rates = regions.groupby('region_group')['is_two'].mean()
        region_table = pd.crosstab(
            regions['region_group'],
            regions['is_two']
        )
        region_p_value = (
            chi2_contingency(region_table)[1]
            if region_table.shape == (2, 2)
            else float('nan')
        )

        return {
            'event_return_rates': event_rates,
            'event_p_value': event_p_value,
            'region_return_rates': region_rates,
            'region_p_value': region_p_value
        }

    def correlation_analysis(self):
        profiles = self.create_user_profiles()

        columns = [
            'total_orders',
            'avg_revenue_rub',
            'avg_days_between_orders',
            'first_device',
            'first_region',
            'first_event_type',
            'first_weekday_name'
        ]

        profile_corr = profiles[columns].copy()

        return profile_corr.phik_matrix(
            interval_cols=[
                'total_orders',
                'avg_revenue_rub',
                'avg_days_between_orders'
            ]
        )