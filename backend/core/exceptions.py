from fastapi import HTTPException


class SCFBaseException(Exception):
    def __init__(self, message: str, detail: dict | None = None):
        self.message = message
        self.detail = detail or {}
        super().__init__(message)


class InvoiceValidationError(SCFBaseException):
    pass


class DuplicateInvoiceError(SCFBaseException):
    pass


class GraphQueryError(SCFBaseException):
    pass


class ModelInferenceError(SCFBaseException):
    pass


class RAGRetrievalError(SCFBaseException):
    pass


class FileParsingError(SCFBaseException):
    pass


class EntityNotFoundError(SCFBaseException):
    pass


def invoice_not_found(invoice_id: str):
    raise HTTPException(status_code=404, detail=f"Invoice {invoice_id} not found")


def fraud_check_failed(reason: str):
    raise HTTPException(status_code=422, detail=f"Fraud check failed: {reason}")
