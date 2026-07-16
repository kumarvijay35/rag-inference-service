"""
UPDATED VIEWS — drop-in replacements for exactly THREE views in chatbot/views.py:
  1. DocumentUploadView
  2. DocumentDeleteView
  3. AskQuestionView

Everything else (auth, sessions, list, history) stays UNCHANGED.

New imports to add at the top of views.py:
    from . import inference_client
    from .inference_client import InferenceServiceError
    from .text_extraction import extract_text, TextExtractionError

Imports/lines that become DEAD after this change:
    from .rag_service import ingest_document, answer_question, delete_document_vectors
"""

import os

from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.permissions import IsAuthenticated

from .models import Document, ChatSession, ChatMessage
from . import inference_client
from .inference_client import InferenceServiceError
from .text_extraction import extract_text, TextExtractionError


@method_decorator(csrf_exempt, name='dispatch')
class DocumentUploadView(APIView):
    permission_classes = [IsAuthenticated]
    parser_classes     = [MultiPartParser, FormParser]

    def post(self, request):
        file = request.FILES.get('file')
        if not file:
            return Response({"error": "No file provided"}, status=400)

        ext = os.path.splitext(file.name)[1].lower()
        if ext not in ['.pdf', '.txt']:
            return Response({"error": "Only PDF and TXT supported"}, status=400)

        doc = Document.objects.create(
            user=request.user,
            name=file.name,
            file=file
        )

        try:
            # Extraction stays in Django (file handling is Django's job);
            # chunking + embedding + indexing are delegated to the
            # FastAPI inference service.
            text   = extract_text(doc.file.path)
            result = inference_client.embed_document(
                user_id=request.user.id,
                document_id=doc.id,
                text=text,
                filename=doc.name,
            )
            doc.chunk_count = result["chunks_indexed"]
            doc.save()
        except TextExtractionError as e:
            doc.delete()
            return Response({"error": str(e)}, status=400)
        except InferenceServiceError:
            doc.delete()
            return Response(
                {"error": "Indexing service is temporarily unavailable, try again shortly"},
                status=503
            )

        return Response({
            "id":      str(doc.id),
            "name":    doc.name,
            "chunks":  doc.chunk_count,
            "message": "Document uploaded and indexed"
        }, status=201)


@method_decorator(csrf_exempt, name='dispatch')
class DocumentDeleteView(APIView):
    permission_classes = [IsAuthenticated]

    def delete(self, request, document_id):
        try:
            doc = Document.objects.get(id=document_id, user=request.user)
        except Document.DoesNotExist:
            return Response({"error": "Not found"}, status=404)

        try:
            inference_client.delete_document(request.user.id, doc.id)
        except InferenceServiceError:
            # Don't block the user's delete if the index service is down;
            # orphaned vectors are filtered out by ownership anyway.
            pass

        if doc.file and os.path.exists(doc.file.path):
            os.remove(doc.file.path)
        doc.delete()
        return Response({"message": "Document deleted"})


@method_decorator(csrf_exempt, name='dispatch')
class AskQuestionView(APIView):
    """
    POST /api/ask/
    Body: { "session_id": "...", "question": "..." }
    Searches across all documents in the session (delegated to the
    FastAPI inference service).
    """
    permission_classes = [IsAuthenticated]

    def post(self, request):
        session_id = request.data.get('session_id')
        question   = request.data.get('question', '').strip()

        if not session_id or not question:
            return Response({"error": "session_id and question required"}, status=400)

        try:
            session = ChatSession.objects.get(id=session_id, user=request.user)
        except ChatSession.DoesNotExist:
            return Response({"error": "Session not found"}, status=404)

        # Ownership check stays in Django: only this user's session's docs
        session_docs = list(session.documents.all())
        doc_ids   = [d.id for d in session_docs]
        doc_names = {str(d.id): d.name for d in session_docs}

        try:
            result = inference_client.query_documents(
                user_id=request.user.id,
                question=question,
                document_ids=doc_ids,
            )
        except InferenceServiceError:
            return Response(
                {"error": "Answering service is temporarily unavailable, try again shortly"},
                status=503
            )

        # Attach human-readable document names to sources
        sources = [
            {
                "document": doc_names.get(s["document_id"], "Unknown"),
                "snippet":  s["text"][:300],
                "score":    s["score"],
            }
            for s in result["sources"]
        ]

        ChatMessage.objects.create(
            session=session,
            question=question,
            answer=result["answer"],
            sources=sources
        )

        return Response({
            "question": question,
            "answer":   result["answer"],
            "sources":  sources
        })
