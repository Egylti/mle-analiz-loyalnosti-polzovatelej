from src.processing import DataProcessor
processor = DataProcessor(
    'data/orders.csv',
    'data/final_tickets_tenge_df.csv'
)
processor.load_data()
data = processor.preprocess_data()
profiles = processor.create_user_profiles()
retention = processor.calculate_retention()
hypotheses = processor.check_hypotheses()
correlations = processor.correlation_analysis()

print('\nИтоги')
print(f'Заказов: {len(data):,}')
print(f'Пользователей: {len(profiles):,}')
print(f'Повторные покупатели: {profiles["is_repeat"].mean():.1%}')

print('\nПовторные покупки по группам')
print(retention.to_string(index=False))

print('\nДоли повторных покупок: спорт и концерты')
print(hypotheses['event_return_rates'])
print(f'p-value: {hypotheses["event_p_value"]:.4f}')
print(f'p-value для регионов: {hypotheses["region_p_value"]:.4f}')

print('\nКорреляции')
print(correlations.round(3))