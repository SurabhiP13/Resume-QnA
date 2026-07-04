from pathlib import Path
from tqdm import tqdm
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

def convert_pdfs_to_markdown(pdf_dir: Path, output_dir: Path, max_resumes: int = 200) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_files = list(pdf_dir.glob("*.pdf"))[:max_resumes]
    if not pdf_files:
        raise ValueError(f"No PDF files found in {pdf_dir}")
    # Resumes are digital text PDFs, not scans; OCR is unneeded and the
    # installed RapidOCR/torch build is incompatible with this docling version.
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )
    ok, failed = 0, []
    for pdf in tqdm(pdf_files, desc="PDF → Markdown"):
        try:
            result = converter.convert(str(pdf))
            md = result.document.export_to_markdown()
            (output_dir / f"{pdf.stem}.md").write_text(md, encoding="utf-8")
            ok += 1
        except Exception as e:
            failed.append((pdf.name, str(e)))
    if failed:
        print(f"Failed ({len(failed)}): " + ", ".join(n for n, _ in failed[:5]) + (" ..." if len(failed) > 5 else ""))
    return ok