import boto3
import PyPDF2
import json
from typing import List, Dict


class PDFProcessor:
    def __init__(self, aws_region: str):
        """Initialize the PDF processor with AWS credentials"""
        self.bedrock = boto3.client(
            service_name='bedrock-runtime',
            region_name=aws_region
        )

    def read_pdf(self, file_path: str) -> str:
        """Read and extract text from PDF file"""
        try:
            with open(file_path, 'rb') as file:
                # Create PDF reader object
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""

                # Extract text from each page
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"

                return text
        except Exception as e:
            raise Exception(f"Error reading PDF: {str(e)}")

    def create_claude_prompt(self, context: str, query: str) -> Dict:
        """Create the prompt for Claude in the required format"""
        return {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1000,
            "messages": [
                {
                    "role": "user",
                    "content": [

                      {
                        "type": "text",
                        "text": f"Context: {context}\n\nQuery: {query}"
                      }
                    ]
                }
            ]
        }

    def query_claude(self, pdf_text: str, user_query: str) -> str:
        """Send query to Claude through AWS Bedrock"""
        # try:
        # Prepare the prompt
        prompt = self.create_claude_prompt(pdf_text, user_query)

        # Convert prompt to JSON string
        body = json.dumps(prompt)

        # Make the API call
        response = self.bedrock.invoke_model(
            modelId="anthropic.claude-3-5-sonnet-20240620-v1:0",
            body=body
        )

        # Parse and return the response
        response_body = json.loads(response.get('body').read())
        return response_body['content'][0]['text']

        # except Exception as e:
        #     raise Exception(f"Error querying Claude: {str(e)}")


def main():
    # Initialize the processor
    processor = PDFProcessor(aws_region='us-east-1')  # Change region as needed

    # Get PDF path from user
    pdf_path = input("Enter the path to your PDF file: ")

    try:
        # Read the PDF
        print("Reading PDF...")
        pdf_text = processor.read_pdf(pdf_path)
        print("PDF successfully loaded!")

        # Interactive query loop
        while True:
            # Get query from user
            query = input("\nEnter your query (or 'quit' to exit): ")

            if query.lower() == 'quit':
                break

            # Process query
            print("\nProcessing your query...")
            response = processor.query_claude(pdf_text, query)
            print("\nClaude's Response:")
            print(response)

    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()