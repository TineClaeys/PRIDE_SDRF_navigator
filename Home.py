import streamlit as st
import requests
import plotly.express as px
import pandas as pd
import io
from ftplib import FTP
import os
import subprocess
#set streamlit wide
st.set_page_config(layout="wide")


# Function to get the SDRF annotation count
def fetch_data():
    sdrfs_url = "https://www.ebi.ac.uk/pride/ws/archive/v2/search/projects?keyword=sdrf"
    all_pxds_url = "https://www.ebi.ac.uk/pride/ws/archive/v2/search/projects?"
    
    number_sdrfs = int(requests.get(sdrfs_url).headers['total_records'])
    number_pxds = int(requests.get(all_pxds_url).headers['total_records'])
    
    return number_sdrfs, number_pxds

# Function to create the donut chart
def plot_donut_chart(number_sdrfs, number_pxds):
    annotated = number_sdrfs
    not_annotated = number_pxds - number_sdrfs
    labels = ['SDRF-annotated', 'Not annotated']
    sizes = [annotated, not_annotated]
    colors = ['orange', 'lightgrey']

    # Create the donut chart using Plotly
    fig = px.pie(
        names=labels,
        values=sizes,
        color=labels,
        color_discrete_map={"SDRF-annotated": "orange", "Not annotated": "lightgrey"},
        hole=0.4,
        labels={'labels': 'Status'},
        title=f"PRIDE Projects with SDRF Annotation: {annotated}/{number_pxds}"
    )

    # Update chart layout for styling
    fig.update_traces(textinfo='percent+label', pull=[0.1, 0])
    # fig.update_layout(margin=dict(t=20, b=20, l=20, r=20))  # Adjust margins

    # Show the plot in Streamlit
    st.plotly_chart(fig)


# Function to get SDRF FTP link from project accession
def get_sdrf_ftp(pxd, df):
    #first try if it's an indexed SDRF
    #get publicationDate from df
    publication_date = df[df['accession'] == pxd]['publicationDate'].values[0]
    year = publication_date.split("-")[0]
    month = publication_date.split("-")[1]
    for fname in ["sdrf.tsv", "sdrf.txt"]:
        ftp_link = f"ftp://ftp.pride.ebi.ac.uk/pride/data/archive/{year}/{month}/{pxd}/{fname}"
        try:
            sdrf_file = load_sdrf_from_ftp(ftp_link)
            print(f"{fname} found and loaded.")
            return ftp_link
        except Exception as e:
            #this only works if it's a non-indexed SDRF
            files_url = f"https://www.ebi.ac.uk/pride/ws/archive/v2/projects/{pxd}/files"
            response = requests.get(files_url)
            files = response.json()
            for f in files:
                if 'sdrf' in f['fileName'].lower():
                    ftp_link = next((loc['value'] for loc in f['publicFileLocations']
                                    if loc['name'] == 'FTP Protocol'), None)
                    if ftp_link:
                        return ftp_link
            return None

def load_sdrf_from_ftp(ftp_link):
    """
    Download an SDRF file from a given FTP link and return it as a pandas DataFrame.
    Supports .tsv, .txt, .csv, and .xlsx files.
    """
    # Parse FTP path
    ftp_root = "ftp.pride.ebi.ac.uk"
    path_parts = ftp_link.replace(f"ftp://{ftp_root}/", "").split("/")
    directory = "/".join(path_parts[:-1])
    filename = path_parts[-1]
    extension = os.path.splitext(filename)[-1].lower()

    # Connect and download file to memory
    ftp = FTP(ftp_root)
    ftp.login()
    ftp.cwd(directory)

    buffer = io.BytesIO()
    ftp.retrbinary(f"RETR {filename}", buffer.write)
    ftp.quit()
    buffer.seek(0)

    # Read based on file extension
    if extension in [".tsv", ".txt"]:
        return pd.read_csv(buffer, sep="\t", encoding="ISO-8859-1")
    elif extension == ".csv":
        return pd.read_csv(buffer, encoding="ISO-8859-1")
    elif extension == ".xlsx":
        return pd.read_excel(buffer)
    else:
        raise ValueError(f"Unsupported file format: {extension}")

def validate_sdrf(file_path):
    """Runs SDRF validation and returns success status & messages."""
    try:
        result = subprocess.run(
            ["parse_sdrf", "validate-sdrf", "--sdrf_file", file_path],
            capture_output=True,
            text=True,
            check=False  # Prevent exception on failure
        )
        if result.returncode == 0:
            return True, result.stdout.strip()
        else:
            return False, result.stdout.strip()
    except FileNotFoundError:
        return None, "SDRF validation tool not found. Install `sdrf-pipelines` via `pip install sdrf-pipelines`."


st.title("PRIDE SDRF Navigator")
st.text("Interactive overview of PRIDE projects with SDRF annotations for fast access to sample and experimental metadata")

# Fetch the data and show donut chart
number_sdrfs, number_pxds = fetch_data()
plot_donut_chart(number_sdrfs, number_pxds)

full_json = []
page_number = 0
while True:
    response = requests.get(f"https://www.ebi.ac.uk/pride/ws/archive/v2/search/projects?keyword=sdrf&page={page_number}")
    json_data = response.json()
    if len(json_data) == 0:
        break
    else:
        full_json.extend(json_data)
        page_number += 1
#turn json into dataframe
df = pd.DataFrame(full_json)
df = df.drop(columns=['sdrf'])


st.subheader('Metadata of all SDRF annotated PRIDE projects')
search_keyword = st.text_input("Search projects by keyword:")
if search_keyword:
    filtered_df = df[df.apply(lambda row: row.astype(str).str.contains(search_keyword, case=False).any(), axis=1)]
else:
    filtered_df = df

st.dataframe(filtered_df)


# Allow user to select a PXD from the list
st.subheader("Select a Project to retrieve the SDRF file")
st.text("Select a project, if possible to retrieve it will be displayed and searchable below. The righthand column will show the validation status of the SDRF file according to the offical SDRF validator guidelines")

# Allow user to select multiple PXDs from the filtered DataFrame
options = [''] + df['accession'].tolist()

# Make two columns
col1, col2 = st.columns(2)

with col1:
    pxd = st.selectbox("Select Projects (PXD)", options=options)

if pxd:
    ftp_link = get_sdrf_ftp(pxd, df)

    with col2:
        if ftp_link:
            st.success(f"✅ SDRF file found for PXD: {pxd}")
        else:
            st.error("❌ SDRF file could not be found or indexed")

    if ftp_link:
        try:
            sdrf_file = load_sdrf_from_ftp(ftp_link)

            search_keyword2 = st.text_input("Search keyword in SDRF:")
            if search_keyword2:
                filtered_sdrf_file = sdrf_file[sdrf_file.apply(
                    lambda row: row.astype(str).str.contains(search_keyword2, case=False).any(), axis=1)]
            else:
                filtered_sdrf_file = sdrf_file

            st.dataframe(filtered_sdrf_file)

            if not filtered_sdrf_file.empty:
                file_path = "temp_sdrf.tsv"
                filtered_sdrf_file.to_csv(file_path, index=False, sep="\t", encoding="ISO-8859-1")
                is_valid, message = validate_sdrf(file_path)
                #delete the file
                os.remove(file_path)
                with col2:
                    if is_valid is None:
                        st.error(f"❌ Validation failed: {message}")
                    elif is_valid:
                        st.success("✅ SDRF validation passed!")
                    else:
                        st.error("❌ SDRF validation failed.")
            else:
                with col2:
                    st.warning("⚠️ No matches found for the search term.")
        except Exception as e:
            with col2:
                st.error(f"❌ Error loading SDRF file: {e}")
