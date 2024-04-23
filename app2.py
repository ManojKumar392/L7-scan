import fitz

def extract_potential_tables(pdf_path, threshold=0.5):
  """
  Extracts potential tables from a PDF using PyMuPDF.

  Args:
      pdf_path: Path to the PDF document.
      threshold: Confidence threshold (0.0 to 1.0) for filtering bounding boxes.

  Returns:
      A list of dictionaries representing potential tables with coordinates, 
      dimensions, and page number.
  """
  doc = fitz.open(pdf_path)
  potential_tables = []
  num_pages = 0
  for page in doc:
    num_pages += 1

  print(f"Number of pages: {num_pages}")
  for page in doc:
    blocks = page.get_text("blocks")
    for b in blocks:
      # Filter based on block type (check documentation for supported types)
      if b[1] == fitz.TEXT_BLOCK:
        rect = b[4]  # Get bounding box coordinates (x0, y0, x1, y1)
        width, height = rect[2] - rect[0], rect[3] - rect[1]
        aspect_ratio = width / height  # Check for rectangular shape

        # Apply aspect ratio and confidence threshold for potential tables
        if aspect_ratio > 0.5 and aspect_ratio < 2 and b[5] > threshold:
          potential_tables.append({
              "x0": rect[0],
              "y0": rect[1],
              "x1": rect[2],
              "y1": rect[3],
              "width": width,
              "height": height,
              "page_number": page.number + 1  # Page numbers start from 1
          })

  return potential_tables

# Example usage
pdf_path = "C:\\Users\\manoj\\Downloads\\testing1.pdf"
potential_tables = extract_potential_tables(pdf_path)

if potential_tables:
  for table in potential_tables:
    print(f"Table on page {table['page_number']}: ({table['x0']}, {table['y0']}) - ({table['x1']}, {table['y1']})")
else:
  print("No potential tables found in the PDF.")
