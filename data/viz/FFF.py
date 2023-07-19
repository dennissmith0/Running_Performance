# Slice the DataFrame to include only data from 2020 onwards
df_2020_onwards = df.loc['2020-01-01':'2023-07-01']

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(df_2020_onwards.index, df_2020_onwards['Fitness'], label='Fitness')
ax.plot(df_2020_onwards.index, df_2020_onwards['Fatigue'], label='Fatigue')
ax.plot(df_2020_onwards.index, df_2020_onwards['Form'], label='Form = Fitness - Fatigue')

ax.set_xlabel('Time')
ax.set_ylabel('Scores')
ax.set_title('Fitness, Fatigue and Form Over Time')
ax.legend()

plt.show()
