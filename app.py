import streamlit as st
import boto3
import PyPDF2
import json
import os
from typing import Dict


class ApplianceAssistant:
    def __init__(self, aws_region: str):
        """Initialize AWS Bedrock client"""
        self.bedrock = boto3.client(
            service_name='bedrock-runtime',
            region_name=aws_region
        )

    def read_pdf(self, file_path: str) -> str:
        """Read and extract text from PDF file"""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            raise Exception(f"Error reading PDF: {str(e)}")

    def create_claude_prompt(self, context: str, query: str) -> Dict:
        """Create the prompt for Claude"""
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
        try:
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

        except Exception as e:
            raise Exception(f"Error querying Claude: {str(e)}")


def main():
    st.set_page_config(page_title="LG Appliance Assistant", layout="wide")

    st.title("Fix It")
    st.write("Get answers about your Appliance using AI")

    # Initialize session state for storing PDF content
    if 'pdf_content' not in st.session_state:
        st.session_state.pdf_content = {}

    # Dropdown for company (LG only)
    companies = ["LG", "Other"]
    company = st.selectbox(
        "Select Company",
        companies,
        index=0
    )

    appliances = ["Refrigerator", "Other"]
    appliance = st.selectbox(
        "Select Appliance",
        appliances,
        index=0
    )

    # Dropdown for model numbers
    model_numbers = [
        "LTCS20220", "LTCS24223", "LTCS20120", "LTWS24223",
        "LTNS20220", "LTCS20020", "LTCS20030", "LTCS20040",
        "LT57BPSX", "LRTLS2403", "LHTNS2403", "GT66BP",
        "GT58BP", "GT57BP"
    ]

    selected_model = st.selectbox(
        "Select Model Number",
        model_numbers,
        index=0
    )

    # Query input
    user_query = st.text_area(
        "Enter your query about the selected appliance",
        height=100,
        placeholder="Example: What is the recommended temperature setting?"
    )

    # Initialize the assistant
    assistant = ApplianceAssistant(aws_region='us-east-1')

    # Process button
    if st.button("Get Answer"):
        if user_query:
            try:
                with st.spinner("Processing your query..."):
                    if company == "Other" or appliance == "Other":
                        pdf_content = "Use your general knowledge to answer"
                        st.session_state.pdf_content[selected_model] = pdf_content
                    elif selected_model not in st.session_state.pdf_content:
                        # In a real application, you would dynamically fetch the correct PDF
                        # For now, we'll assume PDFs are in a 'manuals' directory
                        # pdf_path = f"manuals/{selected_model}_manual.pdf"
                        pdf_path = f"manuals/LG_Refrigerator.pdf"

                        # Check if file exists
                        if os.path.exists(pdf_path):
                            pdf_content = assistant.read_pdf(pdf_path)
                            st.session_state.pdf_content[selected_model] = pdf_content
                        else:
                            st.error(f"Manual for model {selected_model} not found!")
                            return

                    # Get response from Claude

                    response = assistant.query_claude(
                        st.session_state.pdf_content[selected_model],
                        user_query
                    )

                    # Display response in a nice format
                    st.success("Response received!")
                    st.write("### Answer:")
                    st.write(response)

                    # Add a divider
                    st.divider()

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please enter a query first!")

    # Add some helpful information at the bottom
    with st.expander("Help & Tips"):
        st.write("""
        ### Tips for getting better answers:
        - Be specific in your questions
        - Ask one question at a time
        - Include relevant details like specific features or parts you're asking about

        ### Example questions:
        - What is the recommended temperature setting for the refrigerator?
        - How do I clean the water dispenser?
        - What should I do if the ice maker isn't working?
        """)


if __name__ == "__main__":
    main()