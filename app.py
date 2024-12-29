import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
from PIL import Image
import requests
from io import BytesIO
import base64
# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

pinecone = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

OLLAMA_HOSTED_VM_PUBLIC_IP = os.getenv("OLLAMA_HOSTED_VM_PUBLIC_IP")

class ImageSearchApp:
    def __init__(self):
        self.index = pinecone.Index('songs')
    
    def _encode_image_to_base64(self, image_path):
        if isinstance(image_path, str) and image_path.startswith(('http://', 'https://')):
            response = requests.get(image_path)
            image_data = response.content
        else:
            # If it's a file-like object (from Streamlit uploader)
            if hasattr(image_path, 'read'):
                image_data = image_path.read()
            else:
                # If it's a local file path
                with open(image_path, 'rb') as image_file:
                    image_data = image_file.read()
        
        return base64.b64encode(image_data).decode('utf-8')
    
    def _get_image_description(self, image_source):
        try:
            # Convert image to base64
            base64_image = self._encode_image_to_base64(image_source)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Describe this image in detail, focusing on the main elements, mood, and any notable features. The description should be a paragraph and not bullet points. Also, the idea is that, whatever description you generate will be used to find a matching song, so make the description not too technical and efficient for the use case. DO NOT USE QUOTATION MARKS OR APOSTROPHE IN THE OUTPUT!"
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=300
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"Error getting image description: {str(e)}")
    
    def _generate_embedding(self, text):
        try:
            
            # use ollama
            # Define the endpoint and payload
            url = f"http://{OLLAMA_HOSTED_VM_PUBLIC_IP}/api/embeddings"
            payload = {
                "model": "nomic-embed-text",
                "prompt": text
            }

            # Send the POST request
            response = requests.post(url, json=payload)
            print(response.json())
            return response.json()['embedding']
            
            
        except Exception as e:
            raise Exception(f"Error generating embedding: {str(e)}")
    
    def _query_pinecone(self, embedding, top_k=5):
        try:
            results = self.index.query(
                vector=embedding,
                top_k=top_k,
                include_metadata=True
            )
            return results['matches']
        except Exception as e:
            raise Exception(f"Error querying Pinecone: {str(e)}")

    def process_image(self, image_source):
        try:
            # Get image description directly from the source
            description = self._get_image_description(image_source)
            
            # Generate embedding
            embedding = self._generate_embedding(description)
            
            # Query Pinecone
            results = self._query_pinecone(embedding)
            print(results)
            return {
                'description': description,
                'matches': results
            }
            
        except Exception as e:
            raise Exception(f"Error processing image: {str(e)}")

def main():
    st.set_page_config(page_title="Image to Song Matcher", layout="wide")
    
    st.title("🎵 Image to Song Matcher")
    st.write("Upload an image or provide a URL, and we'll find songs that match its mood and content!")
    
    # Initialize the app
    app = ImageSearchApp()
    
    # Create two columns for input methods
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    
    with col2:
        st.subheader("Or Provide Image URL")
        image_url = st.text_input("Enter image URL")
    
    # Process the image when either method is used
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        
        if st.button("Find Matching Songs (Uploaded Image)"):
            with st.spinner("Processing image..."):
                try:
                    results = app.process_image(uploaded_file)
                    display_results(results)
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    elif image_url:
        try:
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content))
            st.image(image, caption="Image from URL", use_container_width=True)
            
            if st.button("Find Matching Songs (Image URL)"):
                with st.spinner("Processing image..."):
                    try:
                        results = app.process_image(image_url)
                        display_results(results)
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        except Exception as e:
            st.error("Error loading image from URL. Please check the URL and try again.")

def display_results(results):
    st.subheader("Image Description")
    st.write(results['description'])
    
    st.subheader("Matching Songs")
    for i, match in enumerate(results['matches'], 1):
        with st.expander(f"Match {i} (Score: {match.score:.4f})"):
            st.json({
                "google-search": match.metadata['song'] + " by " + match.metadata['artist'],
                "song": match.metadata['song'],
                "artist": match.metadata['artist'],
                "link": match.metadata['link'],
            })

if __name__ == "__main__":
    main()