import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
from streamlit.delta_generator import DeltaGenerator

def visualize_st(
        container:DeltaGenerator,
        df: pd.DataFrame, 
        cols: list[str], 
        choice: str, 
        color: str
    ) -> None:
    with container:
        if len(cols) != 2:
            st.warning("Please select exactly two columns for visualization.")
            return
        x, y = cols
        match(choice):
            case("Line Plot"): st.line_chart(data=df, x=x, y=y, color=color)
            case("Scatter Plot"): st.scatter_chart(data=df, x=x, y=y, color=color)
            case("Bar Plot"): st.bar_chart(data=df, x=x, y=y, color=color)
            case("Map"): st.map(data=df, latitude=x, longitude=y)
            case("Area Chart"): st.area_chart(data=df, x=x, y=y)

def visualize_sns(
    container: DeltaGenerator,
    df: pd.DataFrame,
    cols: list[str],
    choice: str,
    color: str,
    figsize: int,
    grid: bool,
    color_column: str = None
) -> None:
    with container:
        two_col_plots = [
            "Line Plot", "Scatter Plot", "Bar Plot", "Violin Plot",
            "Strip Plot", "Swarm Plot"
        ]
        one_col_plots = ["Histogram", "KDE Plot", "Count Plot"]

        if (choice in two_col_plots and len(cols) != 2) and choice != "Box Plot":
            st.warning("Please select exactly two columns for this plot.")
            return
        if choice in one_col_plots and len(cols) != 1 and choice != "Box Plot":
            st.warning("Please select exactly one column for this plot.")
            return

        code = "import seaborn as sns\n"
        fig, ax = plt.subplots(figsize=(figsize, figsize))

        args={
            "data": df,
            "x": cols[0],
            "y": cols[1] if len(cols) > 1 else None,
            "color": color if color_column is None else None,
            "ax": ax,
            "hue": color_column if color_column is not None else None 
        }

        match choice:
            case "Line Plot":
                sns.lineplot(**args)
                code += f"sns.lineplot(data=df, x='{args['x']}', y='{args['y']}', color='{color}')\n"
            case "Scatter Plot":
                sns.scatterplot(**args)
                code += f"sns.scatterplot(data=df, x='{args['x']}', y='{args['y']}', color='{color}')\n"
            case "Bar Plot":
                sns.barplot(**args)
                code += f"sns.barplot(data=df, x='{args['x']}', y='{args['y']}', color='{color}')\n"
            case "Box Plot":
                if args['y']:
                    sns.boxplot(**args)
                else:
                    sns.boxplot(**args)
                print(repr(args))
                code += f"sns.boxplot(data=df, x='{args['x']}', y='{args['y']}', color='{color}')\n"
            case "Violin Plot":
                sns.violinplot(**args)
                code += f"sns.violinplot(data=df, x='{args['x']}', y='{args['y']}', color='{color}')\n"
            case "Strip Plot":
                sns.stripplot(**args)
                code += f"sns.stripplot(data=df, x='{args['x']}', y='{args['y']}', color='{color}')\n"
            case "Swarm Plot":
                sns.swarmplot(**args)
                code += f"sns.swarmplot(data=df, x='{args['x']}', y='{args['y']}', color='{color}')\n"
            case "Histogram":
                sns.histplot(**args)
                code += f"sns.histplot(data=df, x='{args['x']}', color='{color}')\n"
            case "KDE Plot":
                sns.kdeplot(**args)
                code += f"sns.kdeplot(data=df, x='{args['x']}', color='{color}')\n"
            case "Count Plot":
                sns.countplot(**args)
                code += f"sns.countplot(data=df, x='{args['x']}', color='{color}')\n"
            case _:
                st.warning(f"Plot '{choice}' not implemented in Seaborn.")
                plt.close(fig)
                return

        ax.grid(grid)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        st.pyplot(fig)
        st.text("Code: ")
        st.code(body=code, language="python", line_numbers=True)
        plt.close(fig)

def visualize_plt(
    container: DeltaGenerator,
    df: pd.DataFrame,
    cols: list[str],
    choice: str,
    color: str,
    figsize: int,
    grid: bool
) -> None:
    with container:
        two_col_plots = [
            "Line Plot", "Scatter Plot", "Bar Plot", "Box Plot", "Violin Plot", "Area Plot", "Step Plot", "Hexbin Plot"
        ]
        one_col_plots = ["Histogram"]

        if choice in two_col_plots and len(cols) != 2:
            st.warning("Please select exactly two columns for this plot.")
            return
        if choice in one_col_plots and len(cols) != 1:
            st.warning("Please select exactly one column for this plot.")
            return

        fig, ax = plt.subplots(figsize=(figsize, figsize))
        code = f"import matplotlib.pyplot as plt\nplt.figure(figsize={figsize, figsize})\n"

        match choice:
            case "Line Plot":
                x, y = cols
                ax.plot(df[x], df[y], color=color)
                code += f"plt.plot(df['{x}'], df['{y}'], color='{color}')\n"
            case "Scatter Plot":
                x, y = cols
                ax.scatter(df[x], df[y], color=color)
                code += f"plt.scatter(df['{x}'], df['{y}'], color='{color}')\n"
            case "Bar Plot":
                x, y = cols
                ax.bar(df[x], df[y], color=color)
                code += f"plt.bar(df['{x}'], df['{y}'], color='{color}')\n"
            case "Box Plot":
                x, y = cols
                ax.boxplot([df[x], df[y]], labels=[x, y])
                code += f"plt.boxplot([df['{x}'], df['{y}']], labels=['{x}', '{y}'])\n"
            case "Violin Plot":
                x, y = cols
                ax.violinplot([df[x], df[y]])
                ax.set_xticks([1, 2])
                ax.set_xticklabels([x, y])
                code += f"plt.violinplot([df['{x}'], df['{y}']])\nplt.xticks([1,2], ['{x}', '{y}'])\n"
            case "Histogram":
                x = cols[0]
                ax.hist(df[x], color=color, bins=20)
                code += f"plt.hist(df['{x}'], color='{color}', bins=20)\n"
            case "Area Plot":
                x, y = cols
                ax.fill_between(df[x], df[y], color=color, alpha=0.5)
                code += f"plt.fill_between(df['{x}'], df['{y}'], color='{color}', alpha=0.5)\n"
            case "Step Plot":
                x, y = cols
                ax.step(df[x], df[y], color=color)
                code += f"plt.step(df['{x}'], df['{y}'], color='{color}')\n"
            case "Hexbin Plot":
                x, y = cols
                ax.hexbin(df[x], df[y], gridsize=30, cmap='Blues')
                code += f"plt.hexbin(df['{x}'], df['{y}'], gridsize=30, cmap='Blues')\n"
            case _:
                st.warning(f"Plot '{choice}' not implemented in Matplotlib.")
                plt.close(fig)
                return

        if choice not in ["Box Plot", "Violin Plot", "Histogram", "Hexbin Plot"]:
            ax.set_xlabel(cols[0] if cols else "")
            ax.set_ylabel(cols[1] if len(cols) > 1 else "")
        ax.grid(grid)
        st.pyplot(fig)
        st.text("Code: ")
        st.code(body=code, language="python", line_numbers=True)
        plt.close(fig)

def visualize_plt_categorical(container:DeltaGenerator, df: pd.DataFrame, cols: list[str], choice: str, figsize: int) -> None:
    with container:
        if len(cols) != 1:
            st.warning("Select exactly 1 column")
            return
        
        x = cols[0]
        fig, ax = plt.subplots(figsize=(figsize, figsize))
        match(choice):
            case("Pie Chart"):
                category_counts = df[x].value_counts()
                ax.pie(category_counts, rotatelabels=True, labels=category_counts.index.tolist(), autopct='%1.1f%%')
                st.code(
                    body = f"""category_counts = df[{x}].value_counts() #df is the dataFrame\nplt.pie(category_counts, rotatelabels=True, labels=category_counts.index.tolist(), autopct='%1.1f%%')""", 
                    language="python"
                )
            case("Histogram"): 
                category_counts = df[x].value_counts()
                category_counts.plot(kind='bar')
                st.code(
                    body = f"""category_counts = df[{x}].value_counts()#df is the dataFrame\n
                            category_counts.plot(kind='bar'))""", 
                    language="python"
                )
                plt.xlabel(x)
                plt.ylabel('Frequency') 
        st.pyplot(fig)
        plt.close(fig)

def visualize_plotly(container: DeltaGenerator, df: pd.DataFrame, cols: list[str], choice: str, color_col: str | None, dtype: str, grid: bool) -> None:
    with container:
        fig = None
        code_lines = ["import plotly.express as px"]
        plot_args = {"data_frame": df}
        
        code_string_common_args = "data_frame=df"
        if color_col and color_col in df.columns:
            plot_args["color"] = color_col
            code_string_common_args += f", color='{color_col}'"

        if dtype == "Numerical":
            if choice == "Histogram":
                if not cols or len(cols) != 1:
                    st.warning("Please select exactly one column for Histogram.")
                    return
                x_col = cols[0]
                plot_args["x"] = x_col
                fig = px.histogram(**plot_args)
                code_lines.append(f"fig = px.histogram({code_string_common_args}, x='{x_col}')")
            
            elif choice in ["Line Plot", "Scatter Plot", "Bar Plot"]:
                if len(cols) != 2:
                    st.warning("Please select exactly two columns for these numerical plots.")
                    return
                x_col, y_col = cols
                plot_args["x"] = x_col
                plot_args["y"] = y_col
                code_string_xy_args = f", x='{x_col}', y='{y_col}'"

                if choice == "Line Plot":
                    fig = px.line(**plot_args)
                    code_lines.append(f"fig = px.line({code_string_common_args}{code_string_xy_args})")
                elif choice == "Scatter Plot":
                    fig = px.scatter(**plot_args)
                    code_lines.append(f"fig = px.scatter({code_string_common_args}{code_string_xy_args})")
                elif choice == "Bar Plot":
                    fig = px.bar(**plot_args)
                    code_lines.append(f"fig = px.bar({code_string_common_args}{code_string_xy_args})")
            elif choice == "Box Plot":
                if len(cols) != 2:
                    st.warning("Please select exactly two columns for Box Plot (x: category, y: value).")
                    return
                x_col, y_col = cols
                plot_args["x"] = x_col
                plot_args["y"] = y_col
                fig = px.box(**plot_args)
                code_lines.append(f"fig = px.box({code_string_common_args}, x='{x_col}', y='{y_col}')")
            elif choice == "Violin Plot":
                if len(cols) != 2:
                    st.warning("Please select exactly two columns for Violin Plot (x: category, y: value).")
                    return
                x_col, y_col = cols
                plot_args["x"] = x_col
                plot_args["y"] = y_col
                fig = px.violin(**plot_args)
                code_lines.append(f"fig = px.violin({code_string_common_args}, x='{x_col}', y='{y_col}')")
            elif choice == "Density Heatmap":
                if len(cols) != 2:
                    st.warning("Please select exactly two columns for Density Heatmap.")
                    return
                x_col, y_col = cols
                plot_args["x"] = x_col
                plot_args["y"] = y_col
                fig = px.density_heatmap(**plot_args)
                code_lines.append(f"fig = px.density_heatmap({code_string_common_args}, x='{x_col}', y='{y_col}')")
            else:
                st.warning(f"Plot choice '{choice}' not yet implemented for Plotly Numerical.")
                return

        elif dtype == "Categorical":
            if not cols or len(cols) != 1:
                st.warning("Please select exactly one column for categorical plots.")
                return
            cat_col = cols[0]
            
            if choice == "Bar Plot": # Count plot
                plot_args["x"] = cat_col
                fig = px.histogram(**plot_args) # px.histogram for counts
                code_lines.append(f"fig = px.histogram({code_string_common_args}, x='{cat_col}') # Counts occurrences")
            elif choice == "Pie Chart":
                counts_df = df[cat_col].value_counts().reset_index()
                counts_df.columns = [cat_col, 'count']
                fig = px.pie(data_frame=counts_df, names=cat_col, values='count', title=f'Distribution of {cat_col}')
                code_lines.append(f"counts_df = df['{cat_col}'].value_counts().reset_index()")
                code_lines.append(f"counts_df.columns = ['{cat_col}', 'count']")
                code_lines.append(f"fig = px.pie(data_frame=counts_df, names='{cat_col}', values='count', title='Distribution of {cat_col}')")
            elif choice == "Sunburst":
                if len(cols) < 1:
                    st.warning("Please select at least one column for Sunburst.")
                    return
                path = cols
                fig = px.sunburst(df, path=path)
                code_lines.append(f"fig = px.sunburst(df, path={path})")
            elif choice == "Treemap":
                if len(cols) < 1:
                    st.warning("Please select at least one column for Treemap.")
                    return
                path = cols
                fig = px.treemap(df, path=path)
                code_lines.append(f"fig = px.treemap(df, path={path})")
            else:
                st.warning(f"Plot choice '{choice}' not yet implemented for Plotly Categorical.")
                return
        else:
            st.error(f"Unknown dtype: {dtype}")
            return

        if fig:
            fig.update_layout(xaxis_showgrid=grid, yaxis_showgrid=grid)
            st.plotly_chart(fig, use_container_width=True)
            st.text("Code (Plotly Express):")
            st.code("\n".join(code_lines), language="python", line_numbers=True)

def visualize_choice(tab:DeltaGenerator) -> None:
    df:pd.DataFrame = st.session_state['visual_df']
    dtype= tab.selectbox(label="Select Dataype", options=["Numerical", "Categorical"])
    
    space1, space2, space3 = tab.columns(spec=3, vertical_alignment="center")
    
    # Plotter options are the same regardless of dtype now, individual functions will handle compatibility
    plotter = space1.radio("Select the Plotter", options=["Matplotlib", "StreamLit", "Seaborn", "Plotly"], key=f"plotter_select_{dtype}")
    
    options = {
        "MatplotlibNumerical": [
            "Line Plot", "Scatter Plot", "Bar Plot", "Box Plot", "Violin Plot",
            "Histogram", "Area Plot", "Step Plot", "Hexbin Plot"
        ],
        "MatplotlibCategorical": [
            "Pie Chart", "Histogram"
        ],
        "StreamLitNumerical": [
            "Line Plot", "Scatter Plot", "Bar Plot", "Area Chart", "Map"
        ],
        "SeabornNumerical": [
            "Line Plot", "Scatter Plot", "Bar Plot", "Box Plot", "Violin Plot",
            "Strip Plot", "Swarm Plot", "Heatmap", "Histogram", "KDE Plot", "Count Plot", "Pair Plot"
        ],
        "PlotlyNumerical": [
            "Line Plot", "Scatter Plot", "Bar Plot", "Histogram", "Box Plot",
            "Violin Plot", "Density Heatmap"
        ],
        "PlotlyCategorical": [
            "Bar Plot", "Pie Chart", "Sunburst", "Treemap"
        ]
    }
    
    plot_options_key = plotter + dtype
    available_choices = options.get(plot_options_key, ["Not available"])

    choice = space2.selectbox(
        label="Select Type of Chart", 
        options=available_choices,
        placeholder="Select plot",
        key=f"chart_choice_{plotter}_{dtype}"
    )
    figsize = space3.number_input(
        label="Enter the size of plot",
        min_value = 10,
        max_value= 100,
        value=10,
        placeholder="Ex: 10",
        key=f"figsize_{plotter}_{dtype}_{choice}",
        help="Primarily for Matplotlib/Seaborn plots."
    )

    col_select_space, color_widget_space, grid_space = tab.columns(spec=3, gap="small", vertical_alignment="center")

    max_cols = 2 
    if plotter == "Plotly":
        if dtype == "Numerical" and choice == "Histogram":
            max_cols = 1
        elif dtype == "Categorical" and choice in ["Bar Plot", "Pie Chart"]:
            max_cols = 1
    elif plotter == "Matplotlib" and dtype == "Categorical": 
        max_cols = 1
    elif plotter == "Seaborn" and choice in ["Pair Plot", "Heatmap"] :
        max_cols = 0

    dtype_options=(list(df.select_dtypes(include="float64")) + list(df.select_dtypes(include="int64"))) if dtype=="Numerical" else list(df.select_dtypes(include="object"))
    
    cols = []
    if not (plotter == "Seaborn" and choice in ["Pair Plot", "Heatmap"]): 
        cols = col_select_space.multiselect(
            label = "Select Columns:",
            options= dtype_options,
            max_selections=max_cols,
            key=f"cols_select_{plotter}_{dtype}_{choice}",
            default=dtype_options[:max_cols] if dtype_options and max_cols > 0 else None
        )

    color_column = None
    picked_color_hex = "#1f77b4" 

    if plotter == "Plotly" or plotter == "Seaborn":
        color_column = color_widget_space.selectbox(
            label="Color by column (optional)",
            options=[None] + list(df.columns),
            key=f"color_col_{plotter}_{dtype}_{choice}",
            help="Select a column to encode color in Plotly charts."
        )
    if plotter in ["StreamLit", "Seaborn", "Matplotlib"]:
        picked_color_hex = color_widget_space.color_picker(
            label="Pick a color:",
            value=picked_color_hex,
            key=f"color_picker_{plotter}_{dtype}_{choice}",
        )
    
    grid = grid_space.checkbox(
        label="Grid",
        value=True,
        key=f"grid_cb_{plotter}_{dtype}_{choice}"
    )
    
    container = tab.container(border=True)
    button_key_base = f"visualize_btn_{plotter}_{dtype}_{choice}"

    if plotter == "StreamLit":
        tab.button(
            label=f"Visualize with Streamlit ({choice})",
            on_click=visualize_st,
            args=(container, df, cols, choice, picked_color_hex),
            use_container_width=True,
            key=f"{button_key_base}_st"
        )
    elif plotter == "Seaborn":
        if choice == "Pair Plot":
            if tab.button(label="Generate Pair Plot (Seaborn)", use_container_width=True, key=f"{button_key_base}_pairplot"):
                with container:
                    if df is not None and not df.empty:
                        st.pyplot(sns.pairplot(data=df, hue=color_column if color_column else None))
                        st.code(body=f"import seaborn as sns\nsns.pairplot(data=df" + (f", hue='{color_column}'" if color_column else "") + ") #df is the dataFrame")
                    else:
                        st.warning("DataFrame is empty or not available.")
        elif choice == "Heatmap":
            if tab.button(label="Generate Heatmap (Seaborn)", use_container_width=True, key=f"{button_key_base}_heatmap"):
                with container:
                    fig, ax = plt.subplots()
                    sns.heatmap(data=df.select_dtypes(exclude="object").corr(), annot=True, cmap="Blues", ax=ax)
                    st.pyplot(fig)
                    st.code(body=f"import seaborn as sns\nsns.heatmap(data=df.select_dtypes(exclude='object').corr(), annot=True, cmap='Blues') #df is the dataFrame")

        else:
            tab.button(
                label=f"Visualize with Seaborn ({choice})",
                on_click=visualize_sns,
                args=(container, df, cols, choice, picked_color_hex, figsize, grid, color_column),
                use_container_width=True,
                key=f"{button_key_base}_sns"
            )
    elif plotter == "Matplotlib":
        if dtype=="Numerical":
            tab.button(
                label=f"Visualize with Matplotlib ({choice})",
                on_click=visualize_plt,
                args=(container, df, cols, choice, picked_color_hex, figsize, grid),
                use_container_width=True,
                key=f"{button_key_base}_plt_num"
            )
        elif dtype=="Categorical":
            tab.button(
                label=f"Visualize with Matplotlib ({choice})",
                on_click=visualize_plt_categorical,
                args=(container, df, cols, choice, figsize),
                use_container_width=True,
                key=f"{button_key_base}_plt_cat"
            )
    elif plotter == "Plotly":
        if choice not in available_choices:
            st.warning(f"Plotly does not support '{choice}' for {dtype} data with current setup.")
        else:
            tab.button(
                label=f"Visualize with Plotly ({choice})",
                on_click=visualize_plotly,
                args=(container, df, cols, choice, color_column, dtype, grid),
                use_container_width=True,
                key=f"{button_key_base}_plotly"
            )
