import boto3
# import pandas as pd
from io import StringIO
from botocore.exceptions import ClientError, NoCredentialsError
from typing import Dict, Optional

# Créez une instance du client S3
s3_client = boto3.client('s3')



def get_s3_object(bucket_name: str, file_key: str, as_utf8: bool) -> Optional[str]:
    
    print('tentative de récupération du fichier '+file_key)
    print('dans le bucket '+bucket_name)
    """
    Retrieve an object from an S3 bucket.

    :param bucket_name: The name of the S3 bucket.
    :param file_key: The key (path) of the file in the S3 bucket.
    :return: The content of the file as a string, or None if an error occurs.
    """
    try:
        # Téléchargez le fichier depuis S3
        response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
        
        # Lisez le contenu du fichier
        if as_utf8 == True:
            file_content = response['Body'].read().decode('utf-8')
        else:
            file_content = response['Body'].read()
            
        return file_content
    
    except ClientError as e:
        # En cas d'erreur, retournez None
        print(f"Erreur lors de la récupération du fichier S3 : {e}")
        return None
        

# def get_acte_as_formated_csv_dataframe(bucket_name: str, file_key: str) -> Optional[pd.DataFrame]:
#     """
#     Retrieve a CSV file from S3, load it into a pandas DataFrame, and return it.

#     :param bucket_name: The name of the S3 bucket.
#     :param file_key: The key (path) of the CSV file in the S3 bucket (without the .csv extension).
#     :return: A pandas DataFrame containing the CSV data, or None if an error occurs.
#     :raises: Propagates exceptions such as NoCredentialsError or other errors to the calling function.
#     """
#     try:

#         response = s3_client.get_object(Bucket=bucket_name, Key=f'{file_key}.csv')
#         content = response['Body'].read().decode('utf-8')
#         df = pd.read_csv(StringIO(content))
        
#         if content is None:
#             raise ValueError("Failed to retrieve CSV content from S3")

#         df = pd.read_csv(StringIO(content))
#         return df

#     except NoCredentialsError as e:
#         print("Credentials not available")
#         raise e
#     except Exception as e:
#         print(f"An error occurred: {e}")
#         raise e
        
        

# def get_formated_page_from_df(df, page_no):
#     """Récupérer et formater le texte d'une page spécifique depuis un dataframe récupéré avec get_formated_document_datas """
#     try:
#         # Filtrer les lignes où PageNo est égal à `page_no`
#         page_data = df[df['PageNo'] == page_no]
#         if not page_data.empty:
#             # Convertir la chaîne en une liste Python
#             text_list = ast.literal_eval(page_data['Text'].values[0])
            
#             # Joindre les éléments de la liste avec des sauts de ligne
#             text_page = "\n".join(text_list)
#             return text_page
#         else:
#             return f"No text found for page number {page_no}"
            
#     except Exception as e:
#         return None