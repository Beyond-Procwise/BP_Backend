"""A throwaway RSA key standing in for our Microsoft app's certificate in tests."""
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from services.mail_m365.graph_client import CertCredentials

KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
PEM = KEY.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                        serialization.NoEncryption()).decode()
PUBLIC_KEY = KEY.public_key()
X5T = "dGh1bWJwcmludA"


def fake_credentials(tenant_id: str = "t", client_id: str = "c") -> CertCredentials:
    return CertCredentials(tenant_id=tenant_id, client_id=client_id,
                           private_key_pem=PEM, x5t=X5T)
