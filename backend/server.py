from fastapi import FastAPI, APIRouter, File, UploadFile, Form, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
import uuid
from datetime import datetime, timezone
import base64
from io import BytesIO
from PIL import Image
from pdf2image import convert_from_bytes
from emergentintegrations.llm.chat import LlmChat, UserMessage, ImageContent


ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")


# Define Models
class StatusCheck(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_name: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class StatusCheckCreate(BaseModel):
    client_name: str

class OCRExtraction(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: str
    extracted_text: str
    prompt_used: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

# Helper function to convert PDF to images
def pdf_to_images(pdf_bytes):
    """Convert PDF bytes to list of PIL images"""
    images = convert_from_bytes(pdf_bytes, dpi=300)
    return images

# Helper function to convert image to base64
def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string"""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

# Add your routes to the router instead of directly to app
@api_router.get("/")
async def root():
    return {"message": "DeepSeek OCR API - Ready"}

@api_router.post("/status", response_model=StatusCheck)
async def create_status_check(input: StatusCheckCreate):
    status_dict = input.model_dump()
    status_obj = StatusCheck(**status_dict)
    
    doc = status_obj.model_dump()
    doc['timestamp'] = doc['timestamp'].isoformat()
    
    _ = await db.status_checks.insert_one(doc)
    return status_obj

@api_router.get("/status", response_model=List[StatusCheck])
async def get_status_checks():
    status_checks = await db.status_checks.find({}, {"_id": 0}).to_list(1000)
    
    for check in status_checks:
        if isinstance(check['timestamp'], str):
            check['timestamp'] = datetime.fromisoformat(check['timestamp'])
    
    return status_checks

@api_router.post("/extract-text")
async def extract_text(
    file: UploadFile = File(...),
    prompt: Optional[str] = Form("Extract all text from this image. Provide the text exactly as it appears, maintaining the original structure and formatting.")
):
    """
    Extract text from uploaded image or PDF file using vision-capable LLM.
    
    Accepts: .jpg, .jpeg, .png, .pdf files
    Returns: Extracted text
    """
    try:
        # Validate file type
        file_ext = file.filename.split('.')[-1].lower()
        if file_ext not in ['jpg', 'jpeg', 'png', 'pdf']:
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload .jpg, .jpeg, .png, or .pdf files.")
        
        # Read file content
        file_content = await file.read()
        
        # Process based on file type
        images = []
        if file_ext == 'pdf':
            # Convert PDF pages to images
            images = pdf_to_images(file_content)
        else:
            # Single image file
            img = Image.open(BytesIO(file_content))
            images = [img]
        
        # Get API key from environment
        api_key = os.environ.get('EMERGENT_LLM_KEY', '')
        if not api_key:
            raise HTTPException(status_code=500, detail="API key not configured")
        
        # Extract text from each image/page
        all_extracted_text = []
        
        for idx, img in enumerate(images):
            # Convert image to base64
            img_base64 = image_to_base64(img)
            
            # Initialize LLM Chat with vision model
            chat = LlmChat(
                api_key=api_key,
                session_id=f"ocr-{uuid.uuid4()}",
                system_message="You are an expert OCR system. Extract text accurately from images."
            ).with_model("openai", "gpt-4o")
            
            # Create image content
            image_content = ImageContent(image_base64=img_base64)
            
            # Create message with image
            user_message = UserMessage(
                text=prompt,
                file_contents=[image_content]
            )
            
            # Get extraction
            response = await chat.send_message(user_message)
            
            if len(images) > 1:
                all_extracted_text.append(f"--- Page {idx + 1} ---\n{response}")
            else:
                all_extracted_text.append(response)
        
        # Combine all extracted text
        final_text = "\n\n".join(all_extracted_text)
        
        # Store in database
        extraction_doc = OCRExtraction(
            filename=file.filename,
            extracted_text=final_text,
            prompt_used=prompt
        )
        
        doc = extraction_doc.model_dump()
        doc['timestamp'] = doc['timestamp'].isoformat()
        
        await db.ocr_extractions.insert_one(doc)
        
        return {
            "success": True,
            "filename": file.filename,
            "extracted_text": final_text,
            "pages_processed": len(images)
        }
        
    except Exception as e:
        logger.error(f"Error extracting text: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@api_router.get("/extractions", response_model=List[OCRExtraction])
async def get_extractions():
    """Get all OCR extraction history"""
    extractions = await db.ocr_extractions.find({}, {"_id": 0}).sort("timestamp", -1).to_list(100)
    
    for extraction in extractions:
        if isinstance(extraction['timestamp'], str):
            extraction['timestamp'] = datetime.fromisoformat(extraction['timestamp'])
    
    return extractions

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()