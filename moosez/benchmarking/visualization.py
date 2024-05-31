import pandas as pd
import numpy as np
import plotly.subplots
import plotly.graph_objects


def add_vrect_to_fig(fig, x, sections):
    """
    Create rectangle areas to visualize sections y-wide in the current `plotly.graph_objects.Figure`.
    Updating dict in-place is much faster than using fig.add_vrect() method.
    https://github.com/plotly/plotly.py/issues/3307
    """
    shapes = []
    unique_sections = pd.unique(sections)

    for section in unique_sections:
        color = plotly.colors.sequential.Rainbow[np.where(unique_sections == section)[0][0]]
        curr_section_ids = (sections == section).to_numpy(dtype=int)
        curr_section_starts_ends = np.diff(curr_section_ids)
        rect_starts = (curr_section_starts_ends > 0)
        rect_ends = (curr_section_starts_ends < 0)
        if sum(rect_starts) < sum(rect_ends):
            rect_starts[0] = True
        elif sum(rect_ends) < sum(rect_starts):
            rect_ends[-1] = True
        rects_x0 = x.to_numpy(dtype=np.float32)[np.append(rect_starts, False)]
        rects_x1 = x.to_numpy(dtype=np.float32)[np.append(False, rect_ends)]
        for ii in np.arange(len(rects_x0)):
            for jj in np.arange(len(fig.data)):
                axis_idx = str(jj+1)
                if axis_idx == "1":
                    axis_idx = ""
                shape = plotly.graph_objs.layout.Shape(
                    x0=rects_x0[ii], x1=rects_x1[ii], y0=0, y1=1,
                    xref=f"x{axis_idx}", yref=f"y{axis_idx} domain",
                    fillcolor=color, line={'width': 0}, opacity=0.25, type="rect")
                shapes.append(shape)
    fig.layout.shapes = shapes


def save_profiler_visu(df, html_filepath="profile_report.html"):
    """Save profiler output in an interactive html plotly report."""
    valid_cols = [col for col in df.columns if (
        (col != "section (str)")
        & (col != "loop_step (str)")
        & (col != "mem_total (GB)")
        & (col != "gpu_mem_total (GB)")
        & (col != "gpu_name (str)"))]
    # subplot creation
    fig = plotly.subplots.make_subplots(
        rows=len(valid_cols), cols=1, shared_xaxes=True,
        vertical_spacing=0.02)
    hovertemplate = [df.columns[ii] + ": %{customdata[" + str(ii) +"]}" for ii, col in enumerate(df.columns)]
    # populate each row with a specific metric
    for ii, col_name in enumerate(valid_cols):
        curr_hovertemplate = hovertemplate.copy()
        col_name_id = [idx for idx, curr_col in enumerate(curr_hovertemplate) if col_name in curr_col][0]
        curr_hovertemplate = ["<b>" + curr_hovertemplate.pop(col_name_id) + "</b>"] + curr_hovertemplate
        curr_hovertemplate = "<br>".join(curr_hovertemplate) + "<extra></extra>"
        fig.add_trace(
            plotly.graph_objects.Scattergl(
                name=col_name, x=df.index, y=df[col_name],
                customdata=df.to_numpy(),
                hovertemplate=curr_hovertemplate,
                mode="lines+markers"), row=ii+1, col=1)
        fig.update_yaxes(title_text=col_name, row=ii+1, col=1)
    # create areas to visualize sections
    add_vrect_to_fig(fig, df.index, df['section (str)'])
    fig.update_layout(
        title_text="Profiler output",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified")
    fig.update_traces(xaxis="x{}".format(str(len(valid_cols))))
    fig.update_xaxes(title_text=df.index.name)
    fig.write_html(html_filepath)


def read_profile_data(filepath):
    """Read profiler data into a pandas Dataframe."""
    return pd.read_csv(filepath, sep="\t", index_col=0)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input", type=str, help="Input filepath to profiling data (*.tsv)")
    args = parser.parse_args()
    profiled_filepath = args.input
    html_filepath = args.input[:-3] + "html"
    tsv_data = read_profile_data(profiled_filepath)
    save_profiler_visu(tsv_data, html_filepath)


if __name__ == '__main__':
    main()
