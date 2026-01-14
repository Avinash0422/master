# streamlit_app.py
import io
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Master", layout="wide")
st.title("Master Processor")

# ----------------- Helpers -----------------
def try_to_date_header(header):
    out = header.copy()
    for i in range(3, len(out)):
        try:
            out[i] = pd.to_datetime(out[i]).strftime("%Y-%m-%d")
        except Exception:
            pass
    return out

def first_nan_idx(seq):
    for i, x in enumerate(seq):
        if isinstance(x, float) and np.isnan(x):
            return i
    return len(seq)

def normalize_parent_child(df_section):
    rows_list = []
    current_parent_id = None
    current_parent_name = None
    for i in range(len(df_section)):
        row = df_section.iloc[i]
        id_value = row['IDs']
        alias_value = row['Alias']
        if pd.notnull(id_value) and id_value != current_parent_id:
            current_parent_id = id_value
            current_parent_name = alias_value
            rows_list.append({'ID': current_parent_id, 'Parent Name': current_parent_name, 'Child Name': None})
        elif pd.notnull(alias_value) and current_parent_id is not None:
            rows_list.append({'ID': current_parent_id, 'Parent Name': current_parent_name, 'Child Name': alias_value})
    return pd.DataFrame(rows_list)

def extract_algo(parent_name: str):
    if not isinstance(parent_name, str) or len(parent_name) < 2:
        return None
    us = parent_name.find("_")
    if us == -1:
        return None
    val = parent_name[1:us]
    return val or None

def tidy_block_as_table(block_df, header):
    block_df = block_df.iloc[:, :len(header)].copy()
    block_df.columns = header
    return block_df

def build_output_for_sheet(df: pd.DataFrame) -> pd.DataFrame:
    mtm_row_index = df[df["Unnamed: 0"] == "MTM"].index[0]
    capital_deployed_row_index = df[df["Unnamed: 0"] == "Capital Deployed"].index[0]
    max_loss_row_index = df[df["Unnamed: 0"] == "Max SL"].index[0]
    AVG_row_index = df[df["Unnamed: 0"] == "AVG %"].index[0]

    mtm_df = df.iloc[mtm_row_index:capital_deployed_row_index + 1].copy()
    capital_deployed_df = df.iloc[capital_deployed_row_index:max_loss_row_index + 1].copy()
    max_loss_df = df.iloc[max_loss_row_index:AVG_row_index + 1].copy()

    for col in range(3, len(mtm_df.columns)):
        mtm_df.iloc[1, col] = mtm_df.iloc[0, col]
    mtm_df = mtm_df.drop(index=mtm_df.index[0]).reset_index(drop=True)
    mtm_df.columns = mtm_df.iloc[0]
    mtm_df = mtm_df.drop(index=0).reset_index(drop=True)

    header = try_to_date_header(mtm_df.columns.tolist())
    mtm_df.columns = header

    if 'IDs' in mtm_df.columns:
        mtm_df['IDs'] = mtm_df['IDs'].fillna(method='ffill')
    if len(mtm_df) > 0:
        mtm_df = mtm_df.iloc[:-1, :]  

    end_index = first_nan_idx(header)
    header = header[:end_index]
    mtm_df = mtm_df.iloc[:, :len(header)].copy()
    mtm_df.columns = header

    new_df = normalize_parent_child(mtm_df)

    capital_deployed_df = capital_deployed_df.iloc[2:].reset_index(drop=True)
    if len(capital_deployed_df) >= 2:
        capital_deployed_df = capital_deployed_df.drop(
            index=[capital_deployed_df.index[-2], capital_deployed_df.index[-1]]
        ).reset_index(drop=True)
    capital_deployed_df = tidy_block_as_table(capital_deployed_df, header)
    if 'IDs' in capital_deployed_df.columns:
        capital_deployed_df['IDs'] = capital_deployed_df['IDs'].fillna(method='ffill')

    max_loss_df = tidy_block_as_table(max_loss_df, header)
    max_loss_df = max_loss_df.iloc[2:].reset_index(drop=True)
    if len(max_loss_df) >= 2:
        max_loss_df = max_loss_df.drop(
            index=[max_loss_df.index[-2], max_loss_df.index[-1]]
        ).reset_index(drop=True)
    if 'IDs' in max_loss_df.columns:
        max_loss_df['IDs'] = max_loss_df['IDs'].fillna(method='ffill')

    date_cols = header[3:]
    expanded = pd.DataFrame(new_df.values.repeat(len(date_cols), axis=0), columns=new_df.columns)
    expanded['Date'] = date_cols * len(new_df)
    check = expanded.copy()

    check['Child Name'] = check.apply(
        lambda row: row['Parent Name'] if pd.isna(row['Child Name']) or row['Child Name'] == '' else row['Child Name'],
        axis=1
    )

    check['Capital'] = None
    for idx, row in check.iterrows():
        matched = capital_deployed_df.loc[
            (capital_deployed_df['IDs'] == row['ID']) & 
            (capital_deployed_df['Alias'] == row['Child Name'])
        ]
        if not matched.empty:
            col = row['Date']
            if col in matched.columns:
                val = matched.iloc[0][col]
                if pd.notna(val):
                    check.at[idx, 'Capital'] = val

    check['MTM'] = None
    for idx, row in check.iterrows():
        matched = mtm_df.loc[
            (mtm_df['IDs'] == row['ID']) & 
            (mtm_df['Alias'] == row['Child Name'])
        ]
        if not matched.empty:
            col = row['Date']
            if col in matched.columns:
                val = matched.iloc[0][col]
                if pd.notna(val):
                    check.at[idx, 'MTM'] = val

    check['Max Loss'] = None
    for idx, row in check.iterrows():
        matched = max_loss_df.loc[
            (max_loss_df['IDs'] == row['ID']) & 
            (max_loss_df['Alias'] == row['Child Name'])
        ]
        if not matched.empty:
            col = row['Date']
            if col in matched.columns:
                val = matched.iloc[0][col]
                if pd.notna(val):
                    check.at[idx, 'Max Loss'] = val

    check['algo'] = check['Parent Name'].apply(extract_algo)
    check['Broker'] = "SREDJAINAM2_P"

    final_cols = ['ID', 'Parent Name', 'Child Name', 'Date', 'Capital', 'MTM', 'Max Loss', 'algo', 'Broker']
    check = check[final_cols]

    def is_empty_zero(x):
        if pd.isna(x):
            return True
        if isinstance(x, str) and x.strip() == "":
            return True
        try:
            return float(x) == 0.0
        except Exception:
            return False

    mask_keep = ~check['Capital'].apply(is_empty_zero)
    check = check.loc[mask_keep].reset_index(drop=True)
    return check

# ----------------- Session State -----------------
if "xlsx_bytes" not in st.session_state:
    st.session_state.xlsx_bytes = None
if "sheet_names" not in st.session_state:
    st.session_state.sheet_names = []
if "result" not in st.session_state:
    st.session_state.result = None

# ----------------- Sidebar: upload + process -----------------
with st.sidebar:
    st.header("Steps")
    uploaded = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"], key="uploader")

    if uploaded is not None:
        st.session_state.xlsx_bytes = uploaded.getvalue()
        try:
            xls = pd.ExcelFile(io.BytesIO(st.session_state.xlsx_bytes), engine="openpyxl")
            st.session_state.sheet_names = xls.sheet_names
            st.success(f"Loaded: {len(xls.sheet_names)} sheet(s)")
        except Exception as e:
            st.error(f"Failed to read Excel: {e}")

    if st.session_state.sheet_names:
        default_sheets = [s for s in st.session_state.sheet_names if s.upper().startswith("SEP")] or st.session_state.sheet_names[:1]
        chosen_sheets = st.multiselect(
            "Select sheet(s)",
            options=st.session_state.sheet_names,
            default=default_sheets,
            key="chosen_sheets"
        )
        process_clicked = st.button("Process Sheets")

        if process_clicked:
            try:
                xls = pd.ExcelFile(io.BytesIO(st.session_state.xlsx_bytes), engine="openpyxl")
                outs = []
                for sh in chosen_sheets:
                    df = pd.read_excel(xls, sheet_name=sh, engine="openpyxl")
                    outs.append(build_output_for_sheet(df))
                st.session_state.result = pd.concat(outs, ignore_index=True) if outs else None
                if st.session_state.result is None or st.session_state.result.empty:
                    st.warning("No rows produced.")
                else:
                    st.success("Processed ✅")
            except Exception as e:
                st.error(f"Error while processing: {e}")

# ----------------- Main: tables + filters + cards -----------------
res = st.session_state.result

if res is not None and not res.empty:
    res = res.copy()
    res['Date'] = pd.to_datetime(res['Date'], errors='coerce')
    res['MTM'] = pd.to_numeric(res['MTM'], errors='coerce')

    min_d = pd.to_datetime(res['Date'].min()) if not res.empty else None
    max_d = pd.to_datetime(res['Date'].max()) if not res.empty else None

    if min_d is not None and max_d is not None:
        st.subheader("Date Range")
        c1, c2, c3 = st.columns([1,1,2])
        with c1:
            start_date = st.date_input("Start", value=min_d.date(), min_value=min_d.date(), max_value=max_d.date(), key="start")
        with c2:
            end_date = st.date_input("End", value=max_d.date(), min_value=min_d.date(), max_value=max_d.date(), key="end")

        if start_date > end_date:
            st.info("Start date is after End date → swapping.")
            start_date, end_date = end_date, start_date

        mask = (res['Date'] >= pd.to_datetime(start_date)) & (res['Date'] <= pd.to_datetime(end_date))
        ranged = res.loc[mask].copy().reset_index(drop=True)

        st.markdown(f"**Rows in range:** {len(ranged)} &nbsp;&nbsp; "
                    f"(**From:** {pd.to_datetime(start_date).date()} **To:** {pd.to_datetime(end_date).date()})")

        st.dataframe(ranged, use_container_width=True)

        st.subheader("Stats – Sum of MTM for: VT, RM, GB, RD, PS")
        names = ["VT", "RM", "GB", "RD", "PS"]
        mtm_by_name = (
            ranged[ranged["Child Name"].isin(names)]
            .groupby("Child Name", dropna=False)["MTM"]
            .sum(min_count=1)
            .reindex(names, fill_value=np.nan)
        )
        cols = st.columns(len(names) + 1)
        for i, nm in enumerate(names):
            val = mtm_by_name.loc[nm]
            cols[i].metric(f"{nm} MTM Sum", f"{val:,.2f}" if pd.notna(val) else "—")
        total_val = mtm_by_name.sum(skipna=True)
        cols[-1].metric("Total MTM Sum", f"{total_val:,.2f}" if pd.notna(total_val) else "—")

        dl_main = ranged.copy()
        dl_main['Date'] = dl_main['Date'].dt.strftime("%Y-%m-%d")
        st.download_button(
            label="Download CSV (Selected Date Range – Main Table)",
            data=dl_main.to_csv(index=False).encode("utf-8"),
            file_name="processed_jainam_daily_filtered.csv",
            mime="text/csv",
        )

        st.subheader("Partner-format View (VT, RM, GB, RD, PS)")
        partner_keep = ["VT", "RM", "GB", "RD", "PS"]
        partner_df = ranged[ranged["Child Name"].isin(partner_keep)].copy()

        partner_df["Partner"] = partner_df["Child Name"]
        partner_df["User ID"] = partner_df["ID"]
        partner_df["Algo"] = partner_df["algo"]
        partner_df["Allocation"] = partner_df["Capital"]
        partner_df["broker"] = partner_df["Broker"]
        partner_df["dte"] = ""
        partner_df["index"] = ""

        partner_df["Date"] = partner_df["Date"].dt.strftime("%d-%m-%Y")

        partner_view = partner_df[[ 
            "Partner", "Date", "User ID", "Algo", "MTM", "Allocation", "Max Loss", "broker", "dte", "index"
        ]].drop_duplicates(subset=["Partner", "Date", "User ID", "MTM"]).reset_index(drop=True)

        st.dataframe(partner_view, use_container_width=True)

        st.download_button(
            label="Download CSV (Partner-format, Selected Date Range)",
            data=partner_view.to_csv(index=False).encode("utf-8"),
            file_name="partner_format_filtered.csv",
            mime="text/csv",
        )
    else:
        st.info("No valid dates found.")
else:
    st.info("Upload your Excel, choose sheet(s), and click **Process Sheets** to begin.")

st.caption("Tip: The app expects markers 'MTM', 'Capital Deployed', 'Max SL', 'AVG %' in column 'Unnamed: 0'.")
