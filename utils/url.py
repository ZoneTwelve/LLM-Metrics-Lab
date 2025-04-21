from urllib.parse import urlparse, urlunparse, urljoin, urlencode, parse_qsl

def normalize_url(url: str) -> str:
    # Parse the URL into components
    parsed = urlparse(url)

    # Lowercase the scheme and hostname
    scheme = parsed.scheme.lower()
    netloc = parsed.hostname.lower() if parsed.hostname else ''
    if parsed.port:
        netloc += f":{parsed.port}"

    # Normalize path (e.g., remove redundant slashes)
    path = parsed.path or '/'
    path = '/'.join(segment for segment in path.split('/') if segment)  # Remove empty segments
    path = '/' + path

    # Sort query parameters
    query = urlencode(sorted(parse_qsl(parsed.query)))

    # Rebuild the URL
    normalized = urlunparse((scheme, netloc, path, '', query, ''))
    return normalized
