import os
import json
import re
import ast
import math
import requests
import boto3
import base64
import unidecode
import fitz  # PyMuPDF
from io import StringIO
from rapidfuzz import fuzz
from datetime import datetime
from typing import List, Tuple, Dict

from daxte.boto_manager import get_s3_object





def normalize_text(text: str) -> str:
    """
    Normalize a text by converting it to lowercase and removing accents.
    
    Args:
        text (str): The text to normalize.
    
    Returns:
        str: Normalized text.
    """
    return unidecode.unidecode(text.lower())



def calculate_match_score(reference: str, window: str, reference_length: int) -> float:
    # """Calculate the match score between the reference and a window of text."""
    # similarity = SequenceMatcher(None, reference, window).ratio()
    # differences = reference_length * (1 - similarity)
    # similarity_score = similarity * 100
    # difference_penalty = (differences / reference_length) * 50
    # return (similarity_score - difference_penalty) * 2

    res = fuzz.ratio(reference, window)
    return fuzz.ratio(reference, window)

def is_text_in_lines(reference: str, target_lines: List[dict], match_threshold: float = 0.6, level: int = 0) -> Dict[str, List[dict]]:
    """
    Check if the reference text is approximately contained within a sliding window of the target lines.
    This method compares the reference with substrings of combined lines (previous + current + next),
    and returns the matching lines (with Textract data).
    
    Args:
        reference (str): The reference text to find.
        target_lines (List[dict]): List of lines, each containing 'text' and 'datas' (Textract block data).
        match_threshold (float): Minimum match score to consider a match (default 0.7, or 70%).
        level (int): Level to control the depth of matching (default 0).
    
    Returns:
        Dict[str, List[dict]]: Dictionary containing pages and matched lines (with Textract data).
    """

    matched_lines = {}
    matched_pages = []
    
    normalized_reference = normalize_text(reference)
    reference_length = len(normalized_reference)
    window_size = reference_length * 2

    for i, current_line in enumerate(target_lines):
        current_text = current_line.get('text', '')

        previous_line = target_lines[i - 1] if i > 0 else None
        next_line = target_lines[i + 1] if i < len(target_lines) - 1 else None

        previous_text = previous_line.get('text', '') if previous_line else ""
        next_text = next_line.get('text', '') if next_line else ""

        combined_text = f"{previous_text} {current_text} {next_text}".strip()

        if len(combined_text) >= window_size:
            for j in range(len(combined_text) - window_size + 1):
                window = combined_text[j:j + window_size]
                match_score = calculate_match_score(normalized_reference, window, reference_length)

                if match_score >= match_threshold * 100:
                    if level == 1:
                        matched_lines[current_line['datas']['Id']] = current_line
                        if current_line['datas']['Page'] not in matched_pages:
                            matched_pages.append(current_line['datas']['Page'])
                        break
                    else:
                        previous_matches = matched_previous_line = is_text_in_lines(reference, [previous_line], 0.6, 1) if previous_line else {'matched_lines': []}
                        current_matches = matched_current_line = is_text_in_lines(reference, [current_line], 0.6, 1)
                        next_matches = matched_next_line = is_text_in_lines(reference, [next_line], 0.6, 1) if next_line else {'matched_lines': []}

                        for matched in [previous_matches, current_matches, next_matches]:
                            if matched['matched_lines']:
                                matched_line = matched['matched_lines'][0]
                                matched_lines[matched_line['datas']['Id']] = matched_line
                                if matched_line['datas']['Page'] not in matched_pages:
                                    matched_pages.append(matched_line['datas']['Page'])

                        if (previous_line and current_line and next_line and
                            not any(match['matched_lines'] for match in [previous_matches, current_matches, next_matches])):
                            previous_and_current_matches = is_text_in_lines(reference, [previous_line, current_line], 0.6, 1)
                            current_and_next_matches = is_text_in_lines(reference, [current_line, next_line], 0.6, 1)

                            for combined_matches in [previous_and_current_matches, current_and_next_matches]:
                                if combined_matches['matched_lines']:
                                    for matched_line in combined_matches['matched_lines']:
                                        matched_lines[matched_line['datas']['Id']] = matched_line
                                        if matched_line['datas']['Page'] not in matched_pages:
                                            matched_pages.append(matched_line['datas']['Page'])

    return {
        'pages': matched_pages,
        'matched_lines': list(matched_lines.values())
    }


def find_text_blocks_with_tolerance(textract_data: dict, phrases: List[str], tolerance: float = 0.6) -> List[Tuple[dict, str]]:
    """
    Find text blocks in the Textract data that match the phrases with a tolerance and 
    consider multi-line matches.
    
    Args:
        textract_data (dict): The Textract JSON data.
        phrases (List[str]): List of phrases to find.
        tolerance (float): Match threshold (default 0.7, or 70%).
    
    Returns:
        List[Tuple[dict, str]]: A list of tuples with the Textract data and the matched phrase.
    """
    matches = []
    lines = []
    details = {}
    # Récupérer les lignes des blocs Textract
    page_index = 0

    for i in range(len(textract_data)): 
        for block in textract_data[i]['Blocks']:
            if block['BlockType'] == 'PAGE':
                if block['Page'] != page_index:
                    page_index = block['Page']
                
            if block['BlockType'] == 'LINE':
                # Ajouter à chaque ligne l'info Textract dans l'objet 'datas'
                lines.append({'text': normalize_text(block['Text']), 'datas': block})

    count_lines = 0

    print('Recherche des matchs')
    # Vérifier chaque phrase à travers les lignes
    for phrase in phrases:
        print(f'recherche dans les pages de {phrase}')
        matched_lines = is_text_in_lines(phrase, lines, tolerance)
        
        #print(f'Phrase : {phrase}, Pages : {matched_lines['pages']}, matches : {len(matched_lines['matched_lines'])}')
        details[phrase] = {
            'pages': matched_lines['pages'],
        }

        for matched_line in matched_lines['matched_lines']:
            # Ajouter le bloc associé si ce n'est pas déjà dans la liste des résultats
            if matched_line not in matches:
                matches.append((matched_line['datas'], phrase))

    print('recherche des matches terminé')
    return matches, details



def calculate_rotation_offset(page_width: float, page_height: float, rotation_angle: int) -> Tuple[float, float]:
    """
    Calculate the offset caused by rotating a page from landscape to portrait or vice versa.
    
    Args:
        page_width (float): The original width of the page (in pixels).
        page_height (float): The original height of the page (in pixels).
        rotation_angle (int): The angle of rotation (should be 90 or 270 for landscape-to-portrait).
    
    Returns:
        Tuple[float, float]: The calculated offsets for x and y axes.
    """
    if rotation_angle in [90, 270]:
        # Calculate the offset due to rotation
        offset_x = (page_height - page_width) / 2
        offset_y = (page_width - page_height) / 2
        return offset_x, offset_y
    else:
        # No rotation, no offset needed
        return 0, 0

def adjust_bounding_box_for_rotation(bbox, page_width, page_height, rotation_angle):
    """
    Adjust the bounding box coordinates based on the page rotation. The rotation is applied
    around the center of the page, and the bounding box is adjusted accordingly.
    
    Args:
        bbox (dict): The bounding box with 'Left', 'Top', 'Width', and 'Height' keys.
        page_width (float): The width of the page.
        page_height (float): The height of the page.
        rotation_angle (int): The angle of rotation (in degrees, typically 90, 180, or 270).
        
    Returns:
        fitz.Rect: Adjusted rectangle for the bounding box.
    """
    # Convertir les coordonnées de Textract (normalisées entre 0 et 1) en pixels
    left = bbox['Left'] * page_width
    top = bbox['Top'] * page_height
    right = (bbox['Left'] + bbox['Width']) * page_width
    bottom = (bbox['Top'] + bbox['Height']) * page_height

    # Fonction pour appliquer la rotation d'un point autour d'un centre donné
    def rotate_point(x, y, angle, cx, cy):
        radians = math.radians(angle)
        cos_angle = math.cos(radians)
        sin_angle = math.sin(radians)
        x_new = cos_angle * (x - cx) - sin_angle * (y - cy) + cx
        y_new = sin_angle * (x - cx) + cos_angle * (y - cy) + cy
        return x_new, y_new

    # Appliquer la rotation à chaque coin de la boîte autour du centre de la page
    x0, y0 = rotate_point(left, top, rotation_angle, page_width / 2, page_height / 2)
    x1, y1 = rotate_point(right, top, rotation_angle, page_width / 2, page_height / 2)
    x2, y2 = rotate_point(left, bottom, rotation_angle, page_width / 2, page_height / 2)
    x3, y3 = rotate_point(right, bottom, rotation_angle, page_width / 2, page_height / 2)

    # Calculer le rectangle minimum englobant après rotation
    new_left = min(x0, x1, x2, x3)
    new_top = min(y0, y1, y2, y3)
    new_right = max(x0, x1, x2, x3)
    new_bottom = max(y0, y1, y2, y3)

    # Retourner le nouveau rectangle ajusté
    return fitz.Rect(new_left, new_top, new_right, new_bottom)




def draw_highlights(page, matches: List[Tuple[dict, str]], zoom_x: float, zoom_y: float, rotation_angle: int, offset_x: float = 0, offset_y: float = 0):
    """
    Draw highlight boxes around matched phrases on a PDF page with an optional rotation and positional offset.
    
    Args:
        page: The PDF page object.
        matches (List[Tuple[dict, str]]): List of matched text blocks and phrases.
        zoom_x (float): Horizontal zoom factor.
        zoom_y (float): Vertical zoom factor.
        rotation_angle (int): Angle in degrees to rotate the page before applying highlights.
        offset_x (float): Horizontal offset to apply to all bounding boxes (default 0).
        offset_y (float): Vertical offset to apply to all bounding boxes (default 0).
    """

    drawed_lines = {}

    for block, _ in matches:
        
        bbox = block['Geometry']['BoundingBox']

        # Ajuster les coordonnées avec la rotation
        rect = adjust_bounding_box_for_rotation(bbox, page.rect.width, page.rect.height, rotation_angle)
        
        # Appliquer le facteur de zoom
        rect = rect * fitz.Matrix(zoom_x, zoom_y)

        # Appliquer le décalage
        rect.x0 += offset_x
        rect.y0 += offset_y
        rect.x1 += offset_x
        rect.y1 += offset_y

        if block['Id'] not in drawed_lines:

            # Surligner avec une bordure jaune et une legere transparence
            #page.draw_rect(rect, color=(1, 1, 0), fill_opacity=0.3)  # Jaune avec opacité
            page.draw_rect(rect, color=None, fill=(1, 1, 0), fill_opacity=0.3)
            
            drawed_lines[block['Id']] = {
                'count_line_matches': 1,
                'page': block['Page'],
            }
            
        else:
            #La ligne a déja été surlignée : on incrémente juste le compteur
            drawed_lines[block['Id']]['count_line_matches'] += 1
        
        
def highlight_phrases_in_pdf(pdf_data: bytes, matches: List[Tuple[dict, str]], page_number: int, 
                             zoom_x: float = 1.0, zoom_y: float = 1.0, force_rotation_angle: int = 0) -> bytes:
    """
    Highlight the matched phrases on a specific page in a PDF using their positional data, 
    with an optional rotation of the page.
    
    Args:
        pdf_data (bytes): The PDF file data.
        matches (List[Tuple[dict, str]]): List of matched text blocks and phrases.
        page_number (int): The page number to target (0-based index).
        zoom_x (float): Horizontal zoom factor (default 1.0).
        zoom_y (float): Vertical zoom factor (default 1.0).
        rotation_angle (int): Angle in degrees to rotate the page before applying highlights (default 0).
    
    Returns:
        bytes: The modified PDF with highlights.
    """
    
    # Ouvrir le PDF
    pdf_document = fitz.open(stream=pdf_data, filetype="pdf")

    total_pages = pdf_document.page_count
    print(f"Total pages in document: {total_pages}")

    for page_number in range(total_pages):  # Utilisez total_pages ici
        page = pdf_document.load_page(page_number)
        rotation = page.rotation  # Rotation en degrés (0, 90, 180, 270)
        print(f"Page {page_number + 1} has a rotation of {rotation} degrees")


        if force_rotation_angle == 0:
            current_rotation = page.rotation
            
            # Ajuster la rotation en fonction des métadonnées
            if current_rotation == 90:
                # Si la page est en mode paysage (rotation de 90 degrés dans le sens horaire), appliquer une rotation de 270 degrés pour revenir en mode portrait
                rotation_angle = 270
            elif current_rotation == 180:
                # Si la page est à l'envers, appliquer une rotation de 180 degrés
                rotation_angle = 180
            elif current_rotation == 270:
                # Si la page est tournée de 270 degrés (sens antihoraire), appliquer une rotation de 90 degrés dans le sens horaire
                rotation_angle = 90
            else:
                rotation_angle = 0
    
        else:
            rotation_angle = force_rotation_angle
    
        offset_x, offset_y = calculate_rotation_offset(page.rect.width, page.rect.height, rotation_angle = rotation_angle)

        
        # Filtrer les correspondances pour cette page
        page_matches = [match for match in matches if match[0]['Page'] == page_number + 1]  # 'Page' est 1-indexé dans Textract
        #TODO : Peux être ici : limiter les matches
        
        draw_highlights(page, page_matches, zoom_x, zoom_y, rotation_angle, offset_x, offset_y)

        
    
    # Sauvegarder le PDF modifié
    pdf_bytes = pdf_document.write()
    pdf_document.close()
    
    return pdf_bytes































def generate_response(status_code: int, body: dict) -> dict:
    """
    Generates a standardized HTTP response.

    :param status_code: HTTP status code to return.
    :param body: Dictionary containing the response body.
    :return: A dictionary representing the HTTP response.
    """
    print(body)
    return {
        'statusCode': status_code,
        'headers': {
            'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token,X-Amz-User-Agent,X-amz-content-sha256',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'OPTIONS,POST,GET,PUT',
            #'Access-Control-Allow-Credentials': 'true'  # Optional, if needed
        },
        'body': json.dumps(body)
    }
    
    

def get_pdf(key):
    
    bucket_name = 'daxte-bucket-ocr-docs'
    pdf_obj = get_s3_object(bucket_name, key, False)

    return pdf_obj



def get_positionnal_datas(key):
    
    bucket_name = 'daxte-bucket-ocr-docs-texts'
    document_obj = get_s3_object(bucket_name, f'{key}.json', False)
    return document_obj



def lambda_handler(event, context):
    print(event)



    event['requestContext'] = {
        "accountId" : "123456789012",
        "identity"  : {
            "sourceIp" : "",
            "userArn" : ""
        }
    }
    # Gérer les requêtes OPTIONS (CORS preflight)
    if event['httpMethod'] == 'OPTIONS':
        return generate_response(200, {})

    elif event['httpMethod'] == 'GET':
        body = event.get('body', '{}')

        try:
            print('------------------------------------------------')
            print('event body '+str(event['body']))
            query_params = event.get('queryStringParameters', {})
            print(f'query paramas : {query_params}')


            # Récupérer l'acte_id depuis les paramètres de l'URL
            acte_id = event.get('pathParameters', {}).get('acte_id', None)
            print('THE ID IS ' + str(acte_id))  # Exemple d'ID : 62fd6126138a209d5e24a13f

            query_params = event.get('queryStringParameters', {})
            # Récupérer les paramètres de la requête GET
            # Récupérer les valeurs des paramètres et les gérer correctement
            action = query_params.get('action')  # 'read' ou 'download'
            highlight = query_params.get('highlight')  # Chaîne de caractères ou null
            pages = query_params.get('pages')  # "*" ou [1,2,3,4]


            print(f"action: {action}")
            print(f"pages: {pages}")

            # Récupérer l'ID de l'utilisateur à partir des informations Cognito
            account_id = event['requestContext'].get('accountId')
            user_arn = event['requestContext']['identity'].get('userArn')
            source_ip = event['requestContext']['identity'].get('sourceIp')

            # Afficher les valeurs dans les logs
            print(f"Account ID: {account_id}")
            print(f"User ARN: {user_arn}")
            print(f"Source IP: {source_ip}")


            # Si 'highlight' est une chaîne de caractères, la convertir en liste
            if highlight:
                highlight = highlight.split(',')  # Séparer par la virgule pour obtenir une liste
            else:
                print(f"highlight: {highlight}")

            print('')
            print('')
            print('')
            print('')
            print('')
            print('')
            print('')
            print('')
            print('')
            
        except json.JSONDecodeError:
            return generate_response(400, {"error": "Invalid request: Malformed JSON in body."})

        # Clé du PDF à récupérer
        pdf_key = 'INPI/ACTES/20220719/6002/6002.E3.1.0e9515e2-4246-11ed-adba-e96e7e624640/LYd1IRGpKsQu_C0022A1001L522539D20220724H164706TPIJTES003PDBOR.pdf'

        # Générer la clé JSON basée sur la structure du fichier PDF
        if 'INPI_SV2' in pdf_key:
            directory, filename = os.path.split(pdf_key)
            filename_without_ext = os.path.splitext(filename)[0]
            json_key = f"{directory}/{filename_without_ext}/{filename}"
        else:
            json_key = f'{pdf_key}'

        # Chemin temporaire pour stocker les fichiers sous /tmp/<account_id>/
        temp_dir = f"/tmp/{account_id}/"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        # Noms des fichiers
        pdf_data_path = os.path.join(temp_dir, f"{acte_id}_pdf_data")
        textract_data_path = os.path.join(temp_dir, f"{acte_id}_textract_data")

        # Vérifier si les fichiers existent déjà
        if os.path.exists(pdf_data_path) and os.path.exists(textract_data_path):
            print('/*/*/*/*/**/*/*/*/*')
            print('FOUNDED IN MEMORY SUCCESS')
            print('/*/*/*/*/**/*/*/*/*')
            # Charger les données PDF et Textract existantes
            with open(pdf_data_path, 'rb') as pdf_file:
                pdf_data = pdf_file.read()
            
            if highlight and len(highlight) > 0:
                with open(textract_data_path, 'r') as textract_file:
                    textract_data = json.load(textract_file)
        else:
            # Si les fichiers n'existent pas, les générer
            pdf_data = get_pdf(pdf_key)
            # Enregistrer le PDF et les données Textract dans le répertoire temporaire
            with open(pdf_data_path, 'wb') as pdf_file:
                pdf_file.write(pdf_data)

            if highlight and len(highlight) > 0:
                textract_data = json.loads(get_positionnal_datas(json_key))
                with open(textract_data_path, 'w') as textract_file:
                    json.dump(textract_data, textract_file)

        
        if highlight and len(highlight) > 0:
            # Extraire les phrases à surligner
            # phrases_to_highlight = ['arnaud de la bédoyère', 'daxte', 'g3cb', 'société']
            matches, details = find_text_blocks_with_tolerance(textract_data, highlight, tolerance=0.6)
            
            print('Surlignage des matchs dans les pages')
            highlighted_pdf = highlight_phrases_in_pdf(pdf_data, matches, 0, zoom_x=1.0, zoom_y=1.0, force_rotation_angle=0)

            # Encoder le PDF en base64
            encoded_pdf = base64.b64encode(highlighted_pdf).decode('utf-8')
                
        else:
            encoded_pdf = base64.b64encode(pdf_data).decode('utf-8')
        
        
        return generate_response(200, {"type": "pdf","details": details, "datas": encoded_pdf})
