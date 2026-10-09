"""Best-effort URL feature extraction for the live research demo.

This module derives URL lexical features locally and, when requested, fetches a
small amount of HTML from public HTTP(S) pages to derive page-content features.
It deliberately does not query WHOIS, Google, traffic, PageRank, or reputation
services. Unknown features remain NaN and are handled by the model pipeline's
saved median imputer. The resulting URL demo is experimental, not equivalent to
the dataset's fully prepared records or the reported held-out evaluation.
"""
from __future__ import annotations

import ipaddress
import re
import socket
from collections import Counter
from html import unescape
from urllib.parse import urljoin, urlparse, urlunparse

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

from core import feature_columns

SHORTENERS = {
    "bit.ly", "t.co", "tinyurl.com", "goo.gl", "ow.ly", "is.gd", "buff.ly",
    "adf.ly", "bitly.com", "cutt.ly", "rebrand.ly", "shorturl.at", "rb.gy",
    "tiny.cc", "lnkd.in", "s.id", "surl.li", "trib.al", "soo.gd",
}
SUSPICIOUS_TLDS = {
    "zip", "mov", "click", "top", "xyz", "work", "support", "country", "kim",
    "gq", "tk", "ml", "cf", "ga", "rest", "fit", "buzz", "cam", "monster",
}
BRANDS = {
    "paypal", "apple", "google", "microsoft", "amazon", "facebook", "instagram",
    "whatsapp", "netflix", "youtube", "yahoo", "outlook", "office", "icloud",
    "bank", "dropbox", "steam", "adobe", "linkedin", "xfinity",
}
PHISH_HINTS = {
    "secure", "account", "webscr", "login", "signin", "verify", "update", "banking",
    "confirm", "password", "credential", "suspend", "unlock", "wallet", "recover",
    "authenticate", "billing", "support", "alert", "invoice", "payment", "limited",
}


def normalize_url(value: str) -> str:
    value = (value or "").strip()
    if not value:
        raise ValueError("Enter a URL first.")
    if len(value) > 2048:
        raise ValueError("The URL is too long (maximum 2,048 characters).")
    if any(ord(c) < 32 for c in value):
        raise ValueError("The URL contains control characters.")
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", value):
        if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", value):
            raise ValueError("Only HTTP and HTTPS URLs are supported. Enter an address such as https://example.com.")
        value = "https://" + value
    p = urlparse(value)
    if p.scheme.lower() not in {"http", "https"}:
        raise ValueError("Only HTTP and HTTPS URLs are supported.")
    if not p.hostname or " " in p.netloc:
        raise ValueError("Enter a valid web address, such as https://example.com.")
    if p.username is not None or p.password is not None:
        raise ValueError("URLs containing embedded usernames or passwords are not supported.")
    try:
        _ = p.port
    except ValueError as exc:
        raise ValueError("The URL contains an invalid port number.") from exc
    return urlunparse((p.scheme.lower(), p.netloc, p.path or "/", p.params, p.query, ""))


def _words(value: str) -> list[str]:
    return [w.lower() for w in re.findall(r"[a-zA-Z0-9]+", unescape(value)) if w]


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host.strip("[]"))
        return True
    except ValueError:
        return False


def _public_host(host: str) -> bool:
    """Reject local/private/reserved destinations before each outbound request."""
    host = (host or "").rstrip(".").lower()
    if host in {"localhost", "localhost.localdomain"} or host.endswith((".localhost", ".local", ".internal")):
        return False
    try:
        if _is_ip(host):
            addrs = [ipaddress.ip_address(host.strip("[]"))]
        else:
            infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
            addrs = [ipaddress.ip_address(i[4][0]) for i in infos]
        return bool(addrs) and all(a.is_global for a in addrs)
    except (OSError, ValueError):
        return False


def safe_fetch_html(url: str, max_redirects: int = 3, max_bytes: int = 1_000_000):
    """Fetch a small HTML response from a public host without following unsafe redirects.

    Returns (final_url, html_text, redirect_count, note). A failed fetch does not
    prevent URL-only prediction.
    """
    current = url
    session = requests.Session()
    session.trust_env = False  # do not inherit potentially unsafe proxy settings
    headers = {"User-Agent": "PhishGuardResearchDemo/1.0 (educational URL feature extraction)"}
    try:
        for hop in range(max_redirects + 1):
            parsed = urlparse(current)
            if parsed.scheme not in {"http", "https"} or not _public_host(parsed.hostname or ""):
                return current, None, hop, "Page retrieval was blocked because the destination is not a verified public HTTP(S) host."
            # Restrict unusual ports. Website fetching normally requires 80/443.
            if parsed.port and parsed.port not in {80, 443}:
                return current, None, hop, "Page retrieval skipped: only standard HTTP/HTTPS ports are allowed."
            with session.get(current, headers=headers, timeout=(3, 5), allow_redirects=False, stream=True) as resp:
                if resp.is_redirect or resp.is_permanent_redirect:
                    location = resp.headers.get("Location")
                    if not location:
                        return current, None, hop, "The website returned an incomplete redirect."
                    if hop >= max_redirects:
                        return current, None, hop, "The website redirected too many times; only URL features were used."
                    current = urljoin(current, location)
                    continue
                if resp.status_code >= 400:
                    return current, None, hop, f"The website returned HTTP {resp.status_code}; only URL features were used."
                ctype = resp.headers.get("Content-Type", "").lower()
                if ctype and not ("text/html" in ctype or "application/xhtml+xml" in ctype):
                    return current, None, hop, "The URL did not return HTML; only URL features were used."
                chunks, total = [], 0
                for chunk in resp.iter_content(16384):
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > max_bytes:
                        chunks.append(chunk[: max_bytes - (total - len(chunk))])
                        break
                    chunks.append(chunk)
                raw = b"".join(chunks)
                encoding = resp.encoding or "utf-8"
                return current, raw.decode(encoding, errors="replace"), hop, "Page HTML retrieved successfully."
        return current, None, max_redirects, "Page retrieval stopped; only URL features were used."
    except requests.RequestException:
        return current, None, 0, "The page could not be retrieved; URL-only features were used."
    finally:
        session.close()


def extract_url_features(url: str) -> dict:
    """Build the locally derivable URL-syntax subset (no network access)."""
    p = urlparse(url)
    host = (p.hostname or "").lower().rstrip(".")
    path_query = (p.path or "") + ("?" + p.query if p.query else "")
    full = url
    host_labels = [x for x in host.split(".") if x]
    tld = host_labels[-1] if len(host_labels) > 1 else ""
    host_text = host
    path_text = path_query.lower()
    raw_words = _words(full)
    host_words = _words(host_text)
    path_words = _words(path_query)
    words = raw_words or [""]
    host_words = host_words or [""]
    path_words = path_words or [""]
    digit_count = sum(ch.isdigit() for ch in full)
    host_digit_count = sum(ch.isdigit() for ch in host)
    domain_core = host_labels[-2] if len(host_labels) >= 2 else host
    domain_core_tokens = set(_words(domain_core))
    host_tokens = set(host_words)
    path_tokens = set(path_words)
    shortened = any(host == s or host.endswith("." + s) for s in SHORTENERS)
    url_tokens = set(raw_words)
    max_repeat = max((len(m.group(0)) for m in re.finditer(r"(.)\1+", full)), default=1)
    path_suffix = (p.path.rsplit("/", 1)[-1].rsplit(".", 1)[-1].lower() if "." in p.path.rsplit("/", 1)[-1] else "")
    tlds = {"com", "net", "org", "edu", "gov", "mil", "int", "co", "io", "info", "biz", "app", "dev", "ph", "uk", "us"}
    # The dataset's definitions are not all fully specified; these lexical heuristics are estimates.
    result = {
        "length_url": len(full), "length_hostname": len(host), "ip": int(_is_ip(host)),
        "nb_dots": full.count("."), "nb_hyphens": full.count("-"), "nb_at": full.count("@"),
        "nb_qm": full.count("?"), "nb_and": full.count("&"), "nb_eq": full.count("="),
        "nb_underscore": full.count("_"), "nb_tilde": full.count("~"), "nb_percent": full.count("%"),
        "nb_slash": full.count("/"), "nb_star": full.count("*"), "nb_colon": full.count(":"),
        "nb_comma": full.count(","), "nb_semicolumn": full.count(";"), "nb_dollar": full.count("$"),
        "nb_space": full.count(" "), "nb_www": full.lower().count("www"), "nb_com": full.lower().count(".com"),
        "nb_dslash": max(0, full.count("//") - 1),
        "http_in_path": path_text.count("http"),
        "https_token": int("https" in host_text or "https" in path_text),
        "ratio_digits_url": digit_count / max(len(full), 1),
        "ratio_digits_host": host_digit_count / max(len(host), 1),
        "punycode": int("xn--" in host),
        "port": int(bool(p.port and p.port not in {80, 443})),
        "tld_in_path": int(any(re.search(r"(?:^|[./_-])" + re.escape(t) + r"(?:$|[./_-])", path_text) for t in tlds)),
        "tld_in_subdomain": int(any(t in host_labels[:-2] for t in tlds)),
        "abnormal_subdomain": int(len(host_labels) > 3 or (len(host_labels) > 2 and host_labels[0] not in {"www", "m", "mobile", "web"} and len(host_labels[0]) > 20)),
        "nb_subdomains": max(0, len(host_labels) - 1),
        "prefix_suffix": int("-" in domain_core),
        "random_domain": int((sum(c.isdigit() for c in domain_core) / max(len(domain_core), 1) > .25) or (len(domain_core) > 18 and len(set(domain_core)) / max(len(domain_core), 1) > .65)),
        "shortening_service": int(shortened),
        "path_extension": int(bool(path_suffix and path_suffix not in {"html", "htm", "php", "asp", "aspx", "jsp"} and len(path_suffix) <= 8)),
        "nb_redirection": path_text.count("//"), "nb_external_redirection": np.nan,
        "length_words_raw": len(raw_words), "char_repeat": max_repeat,
        "shortest_words_raw": min(map(len, raw_words), default=0),
        "shortest_word_host": min(map(len, host_words), default=0),
        "shortest_word_path": min(map(len, path_words), default=0),
        "longest_words_raw": max(map(len, raw_words), default=0),
        "longest_word_host": max(map(len, host_words), default=0),
        "longest_word_path": max(map(len, path_words), default=0),
        "avg_words_raw": float(np.mean([len(w) for w in raw_words])) if raw_words else 0.0,
        "avg_word_host": float(np.mean([len(w) for w in host_words])) if host_words else 0.0,
        "avg_word_path": float(np.mean([len(w) for w in path_words])) if path_words else 0.0,
        "phish_hints": sum(1 for w in url_tokens if w in PHISH_HINTS),
        "domain_in_brand": int(any(b in domain_core for b in BRANDS)),
        "brand_in_subdomain": int(any(any(b in label for b in BRANDS) for label in host_labels[:-2])),
        "brand_in_path": int(any(any(b in w for b in BRANDS) for w in path_words)),
        "suspecious_tld": int(tld in SUSPICIOUS_TLDS),
        "statistical_report": np.nan,
    }
    return result


def extract_html_features(url: str, html: str, redirects: int = 0) -> dict:
    """Derive observable page-content features from a limited HTML snapshot."""
    p = urlparse(url)
    host = (p.hostname or "").lower()
    def same_site(candidate: str | None) -> bool:
        candidate = (candidate or "").lower().rstrip(".")
        return bool(candidate and (candidate == host or candidate.endswith("." + host)))
    soup = BeautifulSoup(html, "html.parser")
    anchors = soup.find_all("a")
    hrefs = [str(a.get("href") or "").strip() for a in anchors]
    resolved = [urlparse(urljoin(url, h)) for h in hrefs if h]
    int_links = sum(1 for q in resolved if q.hostname and same_site(q.hostname))
    ext_links = sum(1 for q in resolved if q.hostname and not same_site(q.hostname))
    total_links = len(hrefs)
    forms = soup.find_all("form")
    login_form = int(any(f.find("input", attrs={"type": re.compile("password", re.I)}) for f in forms))
    form_actions = [urlparse(urljoin(url, str(f.get("action") or ""))) for f in forms]
    ext_forms = sum(1 for q in form_actions if q.hostname and not same_site(q.hostname))
    stylesheets = [l for l in soup.find_all("link") if "stylesheet" in " ".join(l.get("rel", [])).lower()]
    ext_css = sum(1 for l in stylesheets if (lambda q: q.hostname and not same_site(q.hostname))(urlparse(urljoin(url, str(l.get("href") or "")))))
    icons = [l for l in soup.find_all("link") if any("icon" in str(r).lower() for r in (l.get("rel") or []))]
    ext_favicon = int(any((lambda q: q.hostname and not same_site(q.hostname))(urlparse(urljoin(url, str(l.get("href") or "")))) for l in icons))
    media = soup.find_all(["img", "audio", "video", "source"])
    media_urls = []
    for tag in media:
        val = tag.get("src") or tag.get("data-src") or tag.get("poster")
        if val:
            media_urls.append(urlparse(urljoin(url, str(val))))
    int_media = sum(1 for q in media_urls if q.hostname and same_site(q.hostname))
    ext_media = sum(1 for q in media_urls if q.hostname and not same_site(q.hostname))
    text = str(soup)
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    anchors_safe = sum(1 for h in hrefs if h and h not in {"#", "/#"} and not h.lower().startswith(("javascript:", "data:")))
    all_tags = soup.find_all(["meta", "script", "link"])
    links_in_tags = sum(1 for tag in all_tags for attr in ("href", "src", "content") if tag.get(attr))
    all_refs = [urlparse(urljoin(url, str(v))) for tag in soup.find_all(["script", "link", "img", "iframe"]) for v in [tag.get("src") or tag.get("href") or ""] if v]
    ext_refs = sum(1 for q in all_refs if q.hostname and not same_site(q.hostname))
    total_refs = len(all_refs)
    copyright_text = " ".join((soup.find_all(string=re.compile("copyright|©", re.I)) or [])[:10])
    return {
        "nb_hyperlinks": total_links,
        "ratio_intHyperlinks": int_links / max(total_links, 1),
        "ratio_extHyperlinks": ext_links / max(total_links, 1),
        "nb_extCSS": ext_css,
        "ratio_extRedirection": ext_forms / max(len(forms), 1),
        "ratio_extErrors": np.nan,  # requires attempting assets/navigation targets separately
        "login_form": login_form,
        "external_favicon": ext_favicon,
        "links_in_tags": 100.0 * links_in_tags / max(len(all_tags), 1),
        "ratio_intMedia": 100.0 * int_media / max(len(media_urls), 1),
        "ratio_extMedia": 100.0 * ext_media / max(len(media_urls), 1),
        "iframe": int(bool(soup.find("iframe"))),
        "popup_window": int(bool(re.search(r"window\s*\.\s*open\s*\(", text, re.I))),
        "safe_anchor": 100.0 * anchors_safe / max(total_links, 1),
        "onmouseover": int("onmouseover" in text.lower()),
        "right_clic": int(any(x in text.lower() for x in ["oncontextmenu", "event.button==2", "event.button == 2"])),
        "empty_title": int(not title.strip()),
        "domain_in_title": int(bool(title) and any(w in title.lower() for w in _words(host.replace(".", " ")) if len(w) > 2)),
        "domain_with_copyright": int(any(w in copyright_text.lower() for w in _words(host) if len(w) > 3)),
        "nb_external_redirection": redirects,
    }


def build_feature_frame(url: str, fetch_html: bool = True):
    """Return (feature DataFrame, metadata dict) aligned to the saved 81 predictors."""
    normalized = normalize_url(url)
    features = {f: np.nan for f in feature_columns()}
    features.update(extract_url_features(normalized))
    meta = {"normalized_url": normalized, "final_url": normalized, "fetch_status": "Not attempted", "html_available": False,
            "fetch_note": "HTML retrieval was not requested."}
    if fetch_html:
        final_url, html, redirects, note = safe_fetch_html(normalized)
        meta.update({"final_url": final_url, "fetch_status": "Retrieved" if html is not None else "Unavailable",
                     "html_available": html is not None, "fetch_note": note, "redirects": redirects})
        if html is not None:
            try:
                features.update(extract_html_features(final_url, html, redirects))
            except Exception as exc:
                meta["fetch_note"] = f"HTML was downloaded but could not be analysed ({type(exc).__name__}); URL-only features were used."
    # External lookups are not queried by this prototype; preserve those as unavailable.
    for f in ["whois_registered_domain", "domain_registration_length", "domain_age", "web_traffic", "dns_record", "google_index", "page_rank", "statistical_report"]:
        features[f] = np.nan
    frame = pd.DataFrame([[features.get(c, np.nan) for c in feature_columns()]], columns=feature_columns())
    known = int(frame.iloc[0].notna().sum())
    meta["known_features"] = known
    meta["unknown_features"] = len(feature_columns()) - known
    meta["feature_coverage_pct"] = 100.0 * known / max(len(feature_columns()), 1)
    return frame, meta
