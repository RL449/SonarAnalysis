import scipy.io
import matplotlib.pyplot as plt

# Save figure as PNG
save_path = 'C:/Users/rlessard/Desktop/5593 bda graph/5593_bda graph/0409_bda_low.png'

# Define paths to .mat files
mat_file1 = 'C:/Users/rlessard/Desktop/before_during_after_sonar_output/20240409_updated/before_low_180min.mat'
mat_file2 = 'C:/Users/rlessard/Desktop/before_during_after_sonar_output/20240409_updated/during_low_180min.mat'
mat_file3 = 'C:/Users/rlessard/Desktop/before_during_after_sonar_output/20240409_updated/after_low_180min.mat'

# Load .mat files
data1 = scipy.io.loadmat(mat_file1)
data2 = scipy.io.loadmat(mat_file2)
data3 = scipy.io.loadmat(mat_file3)

# Access variables from files
dissim_before = data1['dissim_col']
impulsivity_before = data1['impulsivity_col']
peakcount_before = data1['peakcount_col']
SPLrms_before = data1['SPLrms_col']
SPLpk_before = data1['SPLpk_col']

dissim_during = data2['dissim_col']
impulsivity_during = data2['impulsivity_col']
peakcount_during = data2['peakcount_col']
SPLrms_during = data2['SPLrms_col']
SPLpk_during = data2['SPLpk_col']

dissim_after = data3['dissim_col']
impulsivity_after = data3['impulsivity_col']
peakcount_after = data3['peakcount_col']
SPLrms_after = data3['SPLrms_col']
SPLpk_after = data3['SPLpk_col']

# Calculate y-axis limits for variables
dissim_min, dissim_max = min(dissim_before.min(), dissim_during.min(), dissim_after.min()), max(dissim_before.max(), dissim_during.max(), dissim_after.max())
impulsivity_min, impulsivity_max = min(impulsivity_before.min(), impulsivity_during.min(), impulsivity_after.min()), max(impulsivity_before.max(), impulsivity_during.max(), impulsivity_after.max())
peakcount_min, peakcount_max = min(peakcount_before.min(), peakcount_during.min(), peakcount_after.min()), max(peakcount_before.max(), peakcount_during.max(), peakcount_after.max())
SPLrms_min, SPLrms_max = min(SPLrms_before.min(), SPLrms_during.min(), SPLrms_after.min()), max(SPLrms_before.max(), SPLrms_during.max(), SPLrms_after.max())
SPLpk_min, SPLpk_max = min(SPLpk_before.min(), SPLpk_during.min(), SPLpk_after.min()), max(SPLpk_before.max(), SPLpk_during.max(), SPLpk_after.max())

# Time points for vertical lines (in minutes)
vertical_lines = [90]

# Plot variables
plt.figure(figsize=(18, 15))

# Before
plt.subplot(5, 3, 1)
plt.plot(dissim_before, color='green')
plt.title('Dissimilarity - Before')
plt.ylabel('Dissimilarity')
plt.ylim(dissim_min, dissim_max)

plt.subplot(5, 3, 4)
plt.plot(impulsivity_before, color='green')
plt.title('Impulsivity - Before')
plt.ylabel('Impulsivity')
plt.ylim(impulsivity_min, impulsivity_max)

plt.subplot(5, 3, 7)
plt.plot(peakcount_before, color='green')
plt.title('Peak Count - Before')
plt.ylabel('Peak Count')
plt.ylim(peakcount_min, peakcount_max)

plt.subplot(5, 3, 10)
plt.plot(SPLrms_before, color='green')
plt.title('SPL RMS - Before')
plt.ylabel('Sound Pressure Level (RMS)')
plt.xlabel('Time (Minutes)')
plt.ylim(SPLrms_min, SPLrms_max)

plt.subplot(5, 3, 13)
plt.plot(SPLpk_before, color='green')
plt.title('SPL pk - Before')
plt.ylabel('Sound Pressure Level (pk)')
plt.xlabel('Time (Minutes)')
plt.ylim(SPLpk_min, SPLpk_max)


# During
plt.subplot(5, 3, 2)
plt.plot(dissim_during, color='red')
for line in vertical_lines:
    plt.axvline(x=line, color='black', linestyle='--')
plt.title('Dissimilarity - During')
plt.ylim(dissim_min, dissim_max)

plt.subplot(5, 3, 5)
plt.plot(impulsivity_during, color='red')
for line in vertical_lines:
    plt.axvline(x=line, color='black', linestyle='--')
plt.title('Impulsivity - During')
plt.ylim(impulsivity_min, impulsivity_max)

plt.subplot(5, 3, 8)
plt.plot(peakcount_during, color='red')
for line in vertical_lines:
    plt.axvline(x=line, color='black', linestyle='--')
plt.title('Peak Count - During')
plt.ylim(peakcount_min, peakcount_max)

plt.subplot(5, 3, 11)
plt.plot(SPLrms_during, color='red')
for line in vertical_lines:
    plt.axvline(x=line, color='black', linestyle='--')
plt.title('SPL RMS - During')
plt.xlabel('Time (Minutes)')
plt.ylim(SPLrms_min, SPLrms_max)

plt.subplot(5, 3, 14)
plt.plot(SPLpk_during, color='red')
for line in vertical_lines:
    plt.axvline(x=line, color='black', linestyle='--')
plt.title('SPL pk - During')
plt.xlabel('Time (Minutes)')
plt.ylim(SPLpk_min, SPLpk_max)


# After
plt.subplot(5, 3, 3)
plt.plot(dissim_after, color='blue')
plt.title('Dissimilarity - After')
plt.ylim(dissim_min, dissim_max)

plt.subplot(5, 3, 6)
plt.plot(impulsivity_after, color='blue')
plt.title('Impulsivity - After')
plt.ylim(impulsivity_min, impulsivity_max)

plt.subplot(5, 3, 9)
plt.plot(peakcount_after, color='blue')
plt.title('Peak Count - After')
plt.ylim(peakcount_min, peakcount_max)

plt.subplot(5, 3, 12)
plt.plot(SPLrms_after, color='blue')
plt.title('SPL RMS - After')
plt.xlabel('Time (Minutes)')
plt.ylim(SPLrms_min, SPLrms_max)

plt.subplot(5, 3, 15)
plt.plot(SPLpk_after, color='blue')
plt.title('SPL pk - After')
plt.xlabel('Time (Minutes)')
plt.ylim(SPLpk_min, SPLpk_max)

plt.tight_layout()

# Save as PNG
plt.savefig(save_path)

# Show the plot
plt.show()
