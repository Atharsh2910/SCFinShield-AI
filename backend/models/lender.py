from backend.models.supplier import Entity


class Lender(Entity):
    __mapper_args__ = {"polymorphic_identity": "lender"}
