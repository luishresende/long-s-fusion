"""Configuration shared by the providers."""
import os

# alternatives kept per position; 0 disables the P(a) - P(b) margin
TOP_K = int(os.environ.get("PROVIDER_TOP_K", "5"))
