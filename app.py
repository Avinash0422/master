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

def find_marker_row(df: pd.DataFrame, marker_variants: list, marker_name: str, sheet_name: str) -> int:
    first_col = df.columns[0]
    col_series = df[first_col].astype(str).str.strip().str.lower()
    
    # 1. Exact match (case-insensitive and stripped)
    for variant in marker_variants:
        variant_clean = variant.lower().strip()
        matches = df[col_series == variant_clean].index
        if len(matches) > 0:
            return matches[0]
            
    # 2. Substring match (case-insensitive)
    for variant in marker_variants:
        variant_clean = variant.lower().strip()
        matches = df[col_series.str.contains(variant_clean, na=False)].index
        if len(matches) > 0:
            return matches[0]
            
    # If not found, raise a helpful error listing available values in the first column
    unique_vals = [str(x).strip() for x in df[first_col].dropna().unique() if str(x).strip()][:40]
    raise ValueError(
        f"In sheet '{sheet_name}': Could not find marker '{marker_name}' (tried variants: {marker_variants}). "
        f"The first column of this sheet has the following values: {unique_vals}"
    )

def build_output_for_sheet(df: pd.DataFrame, sheet_name: str = "") -> pd.DataFrame:
    # 1. Shift dataframe if the label column is not the first column (e.g. column A is empty/margins)
    marker_col_idx = 0
    for col_idx in range(min(4, len(df.columns))):
        col_series = df.iloc[:, col_idx].astype(str).str.strip().str.lower()
        if col_series.str.contains("mtm", na=False).any() or col_series.str.contains("capital", na=False).any():
            marker_col_idx = col_idx
            break
            
    if marker_col_idx > 0:
        df = df.iloc[:, marker_col_idx:].copy()
        
    first_col = df.columns[0]
    col_series = df[first_col].astype(str).str.strip().str.lower()
    
    def find_optional_marker(marker_variants):
        for variant in marker_variants:
            variant_clean = variant.lower().strip()
            # Try exact match
            matches = df[col_series == variant_clean].index
            if len(matches) > 0:
                return matches[0]
            # Try substring
            matches = df[col_series.str.contains(variant_clean, na=False)].index
            if len(matches) > 0:
                return matches[0]
        return None

    # Find the row indices dynamically (MTM and Capital Deployed are required, Max SL and AVG % are optional)
    mtm_row_index = find_optional_marker(["MTM"])
    if mtm_row_index is None:
        unique_vals = [str(x).strip() for x in df[first_col].dropna().unique() if str(x).strip()][:40]
        raise ValueError(f"In sheet '{sheet_name}': Could not find required 'MTM' marker. Values found: {unique_vals}")
        
    capital_deployed_row_index = find_optional_marker(["Capital Deployed", "Capital"])
    if capital_deployed_row_index is None:
        unique_vals = [str(x).strip() for x in df[first_col].dropna().unique() if str(x).strip()][:40]
        raise ValueError(f"In sheet '{sheet_name}': Could not find required 'Capital Deployed' marker. Values found: {unique_vals}")
        
    max_loss_row_index = find_optional_marker(["Max SL", "Max Loss"])
    AVG_row_index = find_optional_marker(["AVG %", "AVG"])

    # Determine boundaries of each section
    if max_loss_row_index is not None:
        capital_deployed_end = max_loss_row_index + 1
    elif AVG_row_index is not None:
        capital_deployed_end = AVG_row_index + 1
    else:
        capital_deployed_end = len(df)
        
    if AVG_row_index is not None:
        max_loss_end = AVG_row_index + 1
    else:
        max_loss_end = len(df)

    mtm_df = df.iloc[mtm_row_index:capital_deployed_row_index + 1].copy()
    capital_deployed_df = df.iloc[capital_deployed_row_index:capital_deployed_end].copy()
    
    if max_loss_row_index is not None:
        max_loss_df = df.iloc[max_loss_row_index:max_loss_end].copy()
    else:
        max_loss_df = None

    # Process MTM Section
    for col in range(3, len(mtm_df.columns)):
        mtm_df.iloc[1, col] = mtm_df.iloc[0, col]
    mtm_df = mtm_df.drop(index=mtm_df.index[0]).reset_index(drop=True)
    mtm_df.columns = mtm_df.iloc[0]
    mtm_df = mtm_df.drop(index=0).reset_index(drop=True)

    header = try_to_date_header(mtm_df.columns.tolist())
    mtm_df.columns = header

    if 'IDs' in mtm_df.columns:
        mtm_df['IDs'] = mtm_df['IDs'].ffill()

    # Clean up non-data rows (sums, headers, nans)
    exclude_labels = {'sum', 'nan', '', 'max sl', 'avg %', 'capital deployed', 'mtm', 'ids'}
    if len(mtm_df) > 0:
        first_col_name = mtm_df.columns[0]
        col_clean = mtm_df[first_col_name].astype(str).str.strip().str.lower()
        mtm_df = mtm_df[~col_clean.isin(exclude_labels)].reset_index(drop=True)

    end_index = first_nan_idx(header)
    header = header[:end_index]
    mtm_df = mtm_df.iloc[:, :len(header)].copy()
    mtm_df.columns = header

    new_df = normalize_parent_child(mtm_df)

    # Process Capital Deployed Section
    capital_deployed_df = capital_deployed_df.iloc[2:].reset_index(drop=True)
    col_clean = capital_deployed_df[first_col].astype(str).str.strip().str.lower()
    capital_deployed_df = capital_deployed_df[~col_clean.isin(exclude_labels)].reset_index(drop=True)
    
    capital_deployed_df = tidy_block_as_table(capital_deployed_df, header)
    if 'IDs' in capital_deployed_df.columns:
        capital_deployed_df['IDs'] = capital_deployed_df['IDs'].ffill()

    # Process Max Loss Section (if present)
    if max_loss_df is not None:
        max_loss_df = max_loss_df.iloc[2:].reset_index(drop=True)
        col_clean = max_loss_df[first_col].astype(str).str.strip().str.lower()
        max_loss_df = max_loss_df[~col_clean.isin(exclude_labels)].reset_index(drop=True)
        
        max_loss_df = tidy_block_as_table(max_loss_df, header)
        if 'IDs' in max_loss_df.columns:
            max_loss_df['IDs'] = max_loss_df['IDs'].ffill()

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
                # Handle duplicate columns
                if isinstance(val, pd.Series):
                    val = val.iloc[0]
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
                # Handle duplicate columns
                if isinstance(val, pd.Series):
                    val = val.iloc[0]
                if pd.notna(val):
                    check.at[idx, 'MTM'] = val

    check['Max Loss'] = None
    if max_loss_df is not None:
        for idx, row in check.iterrows():
            matched = max_loss_df.loc[
                (max_loss_df['IDs'] == row['ID']) & 
                (max_loss_df['Alias'] == row['Child Name'])
            ]
            if not matched.empty:
                col = row['Date']
                if col in matched.columns:
                    val = matched.iloc[0][col]
                    # Handle duplicate columns
                    if isinstance(val, pd.Series):
                        val = val.iloc[0]
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

    mask_capital = ~check['Capital'].apply(is_empty_zero)
    mask_mtm = ~check['MTM'].apply(is_empty_zero)
    
    # Keep row if Capital is valid OR MTM is valid
    mask_keep = mask_capital | mask_mtm
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
        months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
        default_sheets = [
            s for s in st.session_state.sheet_names 
            if any(m in s.upper() for m in months)
        ]
        if not default_sheets:
            default_sheets = st.session_state.sheet_names[:1]
            
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
                    # Debug: Show column names
                    st.info(f"Sheet '{sh}' columns: {list(df.columns)[:5]}")  # Show first 5 columns
                    st.info(f"First column name: '{df.columns[0]}'")
                    
                    # Debug: Show first column values to help identify markers
                    first_col_values = df[df.columns[0]].dropna().unique()[:20]
                    st.info(f"First 20 unique values in first column: {list(first_col_values)}")
                    
                    outs.append(build_output_for_sheet(df, sheet_name=sh))
                st.session_state.result = pd.concat(outs, ignore_index=True) if outs else None
                if st.session_state.result is None or st.session_state.result.empty:
                    st.warning("No rows produced.")
                else:
                    st.success("Processed ✅")
            except Exception as e:
                st.error(f"Error while processing: {e}")
                st.error("Please check that your Excel file has 'MTM', 'Capital Deployed', 'Max SL', 'AVG %' in the first column.")

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
        # Filter for partners and deduplicate for Stats
        stats_df = ranged[ranged["Child Name"].isin(names)].copy()
        # Deduplicate based on: Partner(Child Name), Date, User ID(ID), Algo(algo), MTM
        stats_df = stats_df.drop_duplicates(subset=["Child Name", "Date", "ID", "algo", "MTM"])

        mtm_by_name = (
            stats_df
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
        ]].drop_duplicates(subset=["Partner", "Date", "User ID", "Algo", "MTM"]).reset_index(drop=True)

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
