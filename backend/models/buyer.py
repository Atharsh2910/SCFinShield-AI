from backend.models.supplier import Entity


class Buyer(Entity):
    __mapper_args__ = {"polymorphic_identity": "buyer"}
