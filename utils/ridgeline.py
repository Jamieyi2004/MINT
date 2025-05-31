import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def create_class_distribution_ridgeline(excel_path, output_image_path="class_dist_ridgeline.svg"):
 
    try:
        df_orig = pd.read_excel(excel_path)
    except FileNotFoundError:
        print(f"Error: Excel file not found at '{excel_path}'")
        return
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return

    fixed_domain_order = [
        'tattoo', 'sculpture', 'art', 'toy', 'sticker', 
        'graphic', 'origami', 'painting', 'cartoon', 'embroidery', 
        'graffiti', 'videogame', 'deviantart', 'misc', 'sketch'
    ]
    print(f"Using fixed domain order (top to bottom): {fixed_domain_order}")

    df_orig = df_orig[df_orig['Domain'].isin(fixed_domain_order)]
    df_orig['Domain'] = pd.Categorical(df_orig['Domain'], categories=fixed_domain_order, ordered=True)
    df_orig = df_orig.sort_values('Domain')


    all_unique_classes_sorted = sorted(df_orig['Class'].unique())
    if not all_unique_classes_sorted:
        print("No classes found in the data for the specified domains.")
        return
    class_to_id_map = {cls_name: i for i, cls_name in enumerate(all_unique_classes_sorted)}
    num_total_classes = len(all_unique_classes_sorted) # This should ideally be 200 if all classes are present
    print(f"Found {num_total_classes} unique classes across the specified domains.")
    if num_total_classes != 200:
        print(f"Warning: Expected 200 classes for ImageNet-R, but found {num_total_classes}. X-axis ticks might not align perfectly with a 1-200 scale if data is incomplete.")

    reshaped_data = []
    for domain_name, group in df_orig.groupby('Domain', observed=True):
        for _, row in group.iterrows():
            class_name = row['Class']
            image_count = int(row['Image_Count'])
            if class_name in class_to_id_map:
                class_id = class_to_id_map[class_name]
                for _ in range(image_count):
                    reshaped_data.append({'Domain': domain_name, 'Class_ID': class_id})
    
    if not reshaped_data:
        print("No data to plot after reshaping. Check Image_Counts or domain names.")
        return
    df_plot = pd.DataFrame(reshaped_data)
    df_plot['Domain'] = pd.Categorical(df_plot['Domain'], categories=fixed_domain_order, ordered=True)

    num_plot_domains = len(df_plot['Domain'].unique())
    if num_plot_domains == 0:
        print("No domains to plot based on fixed order after filtering.")
        return

    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0), "figure.facecolor": "white"})
    plt.rcParams['font.family'] = 'sans-serif'
    PREFERRED_SANS_SERIF_FONTS = ['Arial'] # Add your preferred ones
    plt.rcParams['font.sans-serif'] = PREFERRED_SANS_SERIF_FONTS
    try:
        
        c1 = "#4f7dc1" 
        c2 = "#b2c5e5"  
        c3 = "#a6cf8d"  
        c4 = "#75ac50"  
        
        palette = sns.blend_palette([c1, c2, c3, c4], n_colors=num_plot_domains)
        
        # print(f"Using blend_palette from {start_green_low_sat} to {end_blue_low_sat} for {num_plot_domains} domains.")
        domain_to_color_map = {domain: palette[i] for i, domain in enumerate(df_plot['Domain'].unique())}

    except Exception as e:
        print(f"Error setting up palette: {e}")
        palette = sns.color_palette("Greys_r", n_colors=num_plot_domains)
        domain_to_color_map = {domain: palette[i] for i, domain in enumerate(df_plot['Domain'].unique())}

    g = sns.FacetGrid(df_plot, row="Domain", hue="Domain", aspect=15, height=0.6,
                      row_order=df_plot['Domain'].unique().tolist(), 
                      palette=domain_to_color_map,
                      sharex=True)
    
    

    bw_param = 0.4
    g.map_dataframe(sns.kdeplot, x="Class_ID",
                    fill=True, alpha=0.75,
                    clip_on=False, 
                    linewidth=1.0,
                    bw_adjust=bw_param)
    g.map_dataframe(sns.kdeplot, x="Class_ID", color="white", lw=1.5, clip_on=False, bw_adjust=bw_param)
    g.map(plt.axhline, y=0, lw=2, clip_on=False, color='darkgray')

    def annotate_domain_label(data_values_for_column, color, label):
        ax = plt.gca()
        ax.text(-0.015, 0, label, fontweight="bold", color=color,
                ha="right", va="center", transform=ax.transAxes, fontsize=22)

    g.map(annotate_domain_label, "Class_ID")
    
    g.set_titles("")
    for ax in g.axes.flat:
        ax.set_yticks([])
        ax.set_ylabel('')

 
    if num_total_classes == 200: 
 
        custom_tick_labels_1idx = ['1', '50', '100', '150', '200']
        custom_tick_positions_0idx = [0, 49, 99, 149, 199]
        
        g.set(xticks=custom_tick_positions_0idx, xticklabels=custom_tick_labels_1idx)
        plt.xticks(rotation=0, ha='center', fontsize=22)
        g.set_xlabels(f"Class Index", fontsize=26)
    else: # Fallback if class count is not 200
        print(f"Note: Class count is {num_total_classes}, not 200. Using default x-axis ticks or a general rule.")
        # Fallback to a more general rule if class count isn't exactly 200
        tick_interval = 25
        if num_total_classes < 75: tick_interval = 15
        if num_total_classes < 30: tick_interval = 5
        
        tick_positions_0idx = np.arange(0, num_total_classes, tick_interval, dtype=int)
        if (num_total_classes -1) not in tick_positions_0idx and num_total_classes > tick_interval :
             tick_positions_0idx = np.append(tick_positions_0idx, num_total_classes - 1)
        
        tick_labels_1idx = [str(pos + 1) for pos in tick_positions_0idx]
        g.set(xticks=tick_positions_0idx, xticklabels=tick_labels_1idx)
        plt.xticks(rotation=0, ha='center', fontsize=22)
        g.set_xlabels(f"Class Index (1-{num_total_classes})", fontsize=22)

    
    g.despine(bottom=True, left=True)
    plt.subplots_adjust(hspace=-0.7, left=0.2, bottom=0.1, right=0.98, top=0.92)
    # g.fig.suptitle("Class Distribution per Domain", fontsize=14, fontweight="bold")

    try:
        plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
        print(f"Ridgeline plot saved to '{output_image_path}'")
    except Exception as e:
        print(f"Error saving plot: {e}")
    plt.show()

if __name__ == "__main__":
    excel_file_path = 'imagenet_r_distribution.xlsx'
    output_plot_file = 'imagenet_r_class_dist_fixed_ticks.pdf'
    create_class_distribution_ridgeline(excel_file_path, output_plot_file)