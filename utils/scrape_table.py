import pandas as pd

def scrape_table(soup, identifier, col_range=(0, 1000), href_cols=[]):
    """Scrape tabular data from HTML Pages.
    
    Parameters
    ----------
    soup : soup object
        HTML content of web-page
        
    identifier : dict
        Dict containing Class/ID selector for table
        For eg. {'class': 'quarterbacks'} or {'id': 'wide-receivers'}
        Pass only one identifier for a table
        
    col_range : tuple, (int, int)
        Range of columns to be pulled from table
        n refers to Colum (n-1)
        Default value: (0, 1000)
        
    href_cols : list
        Column indices for which the hyperlinked URL also needs to be scraped
        
    Returns
    -------
    df : pandas dataframe
        DataFrame containing all requested columns 
    """
    table = soup.find('table', identifier)
    
    if not table:
        print('Table not found')
        return None
    
    body_rows = table.tbody.find_all('tr')
    data = [[[col.text, col.a.get('href') if col.a else None] if colnum in href_cols else col.text 
             for colnum, col in enumerate(row.find_all(['th','td']))] 
            for row in body_rows]
    
    if table.thead:
        header_rows = table.thead.find_all('tr')
        colnames = [header.text if header.text else f'blank_{colnum}' for colnum, header in enumerate(header_rows[-1].find_all('th'))]
    else:
        colnames = None
    
    df = pd.DataFrame(data).iloc[:,col_range[0]:col_range[1]]
    
    df.columns = colnames if colnames else range(df.shape[1])
    
    return df    
	