import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.ticker as mticker

csv_file_path = '/media/user/Plex/results.csv'
data = pd.read_csv(csv_file_path)
sns.set_theme(style="ticks", palette="pastel")

# filters
tests_to_plot_1 = ['run1_trail2trail', 'run2_along_trail2forest', 'run3_curvy_trail2forest', 'run4_forest_with_obstacles','run5_deep_forest','run7_narrow_trail_forest','run8_big_forest2forest','run9_big_trail2forest'] #test_name
filtered_data_1 = data[data['test_name'].isin(tests_to_plot_1)]
tests_to_plot_2 = ['run1_trail2trail', 'run2_along_trail2forest', 'run3_curvy_trail2forest', 'run4_forest_with_obstacles','run5_deep_forest','run7_narrow_trail_forest'] #test_name
filtered_data_2 = data[data['test_name'].isin(tests_to_plot_2)]

tests_to_plot_big = ['run8_big_forest2forest','run9_big_trail2forest'] #test_name
filtered_data_big = data[data['test_name'].isin(tests_to_plot_big)]

tests_to_plot_3 = ['grass','gravel'] #test_name
filtered_data_3 = data[data['test_name'].isin(tests_to_plot_3)]

tests_to_plot_4 = ['gravel_straight'] #test_name
filtered_data_4 = data[data['test_name'].isin(tests_to_plot_4)]

filtered_data_1['roughness'] = (filtered_data_1['pitch_std_dev'] + filtered_data_1['roll_std_dev']) / 2
filtered_data_2['roughness'] = (filtered_data_2['pitch_std_dev'] + filtered_data_2['roll_std_dev']) / 2
filtered_data_big['roughness'] = (filtered_data_big['pitch_std_dev'] + filtered_data_big['roll_std_dev']) / 2

mean_val = filtered_data_big['total_energy'].mean()
std_val = filtered_data_big['total_energy'].std()
filtered_data_big['normalized_energy'] = (filtered_data_big['total_energy'] - mean_val) / std_val

mean_val = filtered_data_big['total_yaw'].mean()
std_val = filtered_data_big['total_yaw'].std()
filtered_data_big['normalized_yaw'] = (filtered_data_big['total_yaw'] - mean_val) / std_val

max_val = filtered_data_2['total_energy'].max()
min_val = filtered_data_2['total_energy'].min()
filtered_data_2['normalized_energy'] = (filtered_data_2['total_energy'] - min_val) / (max_val - min_val)


####### Energy efficiency plots

# #group boxplot per test
# sns.boxplot(x="terrain_type", y="normalized_energy", hue='type', data=filtered_data_2)
# sns.despine(offset=10, trim=True)


# plt.xlabel("Terrain type")
# plt.ylabel("Total energy - normalized (J)")

# plt.tight_layout()

# #group boxplot per terrain type
# sns.boxplot(x="terrain_type", y="total_energy", hue="type", data=filtered_data_1)
# sns.despine(offset=10, trim=True)



#######Comparison for pitch_std_dev and roll_std_dev for each terrain type

# side-by-side heatmaps of mean values for pitch, roll std_dev, and total_yaw for each type and terrain_type combo
pitch_data = filtered_data_1.pivot_table(index='terrain_type', columns='type', values='pitch_std_dev', aggfunc='mean')
roll_data = filtered_data_1.pivot_table(index='terrain_type', columns='type', values='roll_std_dev', aggfunc='mean')
yaw_data = filtered_data_4.pivot_table(index='test_name', columns='type', values='total_yaw', aggfunc='mean')
yaw_data_rounded = yaw_data.round(1)
rough_data = filtered_data_2.pivot_table(index='terrain_type', columns='type', values='roughness', aggfunc='mean')
fig, (ax4) = plt.subplots(1, 1, figsize=(22, 6))
# sns.heatmap(pitch_data, cmap='YlGnBu', annot=True, ax=ax1)
# ax1.set_title('Pitch Standard Deviation')
# sns.heatmap(roll_data, cmap='YlGnBu', annot=True, ax=ax2)
# ax2.set_title('Roll Standard Deviation')
# sns.heatmap(roll_data, cmap='YlGnBu', annot=True, ax=ax3)
# ax3.set_title('Roughness')
sns.heatmap(yaw_data_rounded, cmap='YlGnBu', annot=True, fmt='.1f', ax=ax4)
ax4.set_title('Total accumulated yaw (degrees)')
ax4.set_xlabel('Navigation type')
ax4.set_ylabel('Terrain type')

# # scatter
# g = sns.FacetGrid(filtered_data_1, col="terrain_type", hue="type", palette="pastel", height=4, aspect=1)
# g.map(sns.scatterplot, "pitch_std_dev", "roll_std_dev")
# g.add_legend(title="Type")
# g.set_axis_labels("Pitch Standard Deviation", "Roll Standard Deviation")
# g.set_titles("Terrain Type: {col_name}")

# # single catplot
# g = sns.catplot(x="type", y="pitch_std_dev", hue="type", col="terrain_type", kind="point", data=filtered_data_1, height=4, aspect=1)
# g.set_axis_labels("Type", "Pitch Standard Deviation")
# g.set_titles("Terrain Type: {col_name}")

# # double catplot
# terrain_types = filtered_data_1['terrain_type'].unique()
# fig, axes = plt.subplots(len(terrain_types), 2, figsize=(15, 5 * len(terrain_types)))
# for idx, terrain_type in enumerate(terrain_types):
#     # Filter data for the specific terrain type
#     terrain_data = filtered_data_1[filtered_data_1['terrain_type'] == terrain_type]

#     # Plot for pitch_std_dev
#     sns.pointplot(x="type", y="pitch_std_dev", hue="type", data=terrain_data, ax=axes[idx, 0])
#     axes[idx, 0].set_ylabel("Pitch Standard Deviation")
#     axes[idx, 0].set_title(f"Terrain Type: {terrain_type}")

#     # Plot for roll_std_dev
#     sns.pointplot(x="type", y="roll_std_dev", hue="type", data=terrain_data, ax=axes[idx, 1])
#     axes[idx, 1].set_ylabel("Roll Standard Deviation")
#     axes[idx, 1].set_title(f"Terrain Type: {terrain_type}")
# plt.tight_layout()

# # 2x nested boxplots
# melted_data = pd.melt(filtered_data_1, id_vars=['type', 'terrain_type'], value_vars=['pitch_std_dev', 'roll_std_dev'], var_name='std_dev_type', value_name='value')
# fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=False)
# for idx, std_dev_type in enumerate(['pitch_std_dev', 'roll_std_dev']):
#     sns.boxplot(x="terrain_type", y="value", hue="type", data=melted_data[melted_data['std_dev_type'] == std_dev_type], ax=axes[idx], palette="pastel")
#     axes[idx].set_ylabel("Standard Deviation")
#     axes[idx].set_title(f"Standard Deviation: {std_dev_type}")
#     axes[idx].set_xlabel("Terrain Type") 
# plt.tight_layout()





# ###### Roughness metric vs. energy efficiency

# # scatter plot
# sns.scatterplot(x="roughness", y="energy_per_odom_meter", hue="type", data=filtered_data_1)
# # Add labels and a title
# plt.xlabel("Roughness")
# plt.ylabel("Energy per Odometer Meter")
# plt.title("Roughness vs. Energy Efficiency by Type")

# # 2D density plot
# g = sns.FacetGrid(filtered_data_1, col="type", height=4, aspect=1)
# g.map_dataframe(sns.kdeplot, x="roughness", y="energy_per_odom_meter", fill=True)
# g.set_axis_labels("Roughness", "Energy per Odometer Meter")
# g.set_titles(col_template="Type: {col_name}")


plt.show()
