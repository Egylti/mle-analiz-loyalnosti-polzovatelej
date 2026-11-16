from src.processing import DataProcessor


processor = DataProcessor(
    'data/orders.csv',
    'data/final_tickets_tenge_df.csv'
)

processor.load_data()
processor.preprocess_data()
profiles = processor.create_user_profiles()
data = processor.data

retention = processor.calculate_retention()
hypotheses = processor.check_hypotheses()
correlations = processor.correlation_analysis()

overall_rate = profiles['is_two'].mean()

print('\nИтоги')
print(f'Заказов после всех фильтров: {len(data):,}')
print(f'Пользователей после фильтрации: {len(profiles):,}')
print(f'Доля повторных покупателей: {overall_rate:.1%}')

print('\nГипотеза: спорт и концерты')
event_rates = hypotheses['event_return_rates']
if {'спорт', 'концерты'}.issubset(event_rates.index):
    print(f"Спорт: {event_rates['спорт']:.1%}")
    print(f"Концерты: {event_rates['концерты']:.1%}")
    print(f"p-value: {hypotheses['event_p_value']:.4g}")
    if hypotheses['event_p_value'] < 0.05:
        print('Разница статистически значима.')
    else:
        print('Убедительной разницы не найдено.')

print('\nГипотеза: топ-10 регионов и остальные')
region_rates = hypotheses['region_return_rates']
if {'Топ-10 регионов', 'Остальные регионы'}.issubset(region_rates.index):
    print(f"Топ-10 регионов: {region_rates['Топ-10 регионов']:.1%}")
    print(f"Остальные регионы: {region_rates['Остальные регионы']:.1%}")
    print(f"p-value: {hypotheses['region_p_value']:.4g}")
    if hypotheses['region_p_value'] < 0.05:
        print('Доли повторных покупок различаются.')
    else:
        print('Убедительной разницы не найдено.')

print('\nСвязь признаков с числом заказов по φK')
print(correlations['total_orders'].drop('total_orders').sort_values(ascending=False).round(3))

segments = retention.loc[
    (retention['users'] >= 30)
    & (retention['return_rate'] > overall_rate)
].sort_values('return_rate', ascending=False).head(3)

if not segments.empty:
    print('\nГруппы для проверки маркетинговых предложений')
    print(segments[['feature', 'segment', 'users', 'return_rate']].to_string(index=False))