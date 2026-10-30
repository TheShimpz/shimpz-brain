//! Text admission rules several Brain inputs share, over the Team protocol crate's Python text semantics.

use team_protocol::python::text;

const LAYOUT: [char; 3] = ['\n', '\r', '\t'];

/// `intent_route._text`: NFC, trimmed, `min..=max` code points, and printable, where `layout` also admits line
/// breaks and tabs (but never another control or format character).
#[must_use]
pub fn admitted(value: &str, max: usize, empty: bool, layout: bool) -> bool {
    let length = value.chars().count();
    if !text::is_nfc(value)
        || text::strip(value) != value
        || length < usize::from(!empty)
        || length > max
    {
        return false;
    }
    if text::isprintable(value) {
        return true;
    }
    let without_layout: String = value
        .chars()
        .filter(|character| !LAYOUT.contains(character))
        .collect();
    if layout && text::isprintable(&without_layout) {
        return true;
    }
    !value.chars().any(|character| {
        let category = text::category(character);
        category.starts_with('C') && (!layout || (category != "Cf" && !LAYOUT.contains(&character)))
    })
}

/// Whether `value` is printable once line feeds and tabs are removed (`str.replace(...).isprintable()`).
#[must_use]
pub fn printable_with_layout(value: &str) -> bool {
    let without: String = value
        .chars()
        .filter(|character| *character != '\n' && *character != '\t')
        .collect();
    text::isprintable(&without)
}

/// Lowercase hexadecimal digits of `bytes`.
#[must_use]
pub fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    bytes
        .iter()
        .flat_map(|byte| {
            [
                DIGITS[usize::from(byte >> 4)],
                DIGITS[usize::from(byte & 15)],
            ]
        })
        .map(char::from)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_admission_follows_the_python_rules() {
        assert!(admitted("Hello", 5, false, false));
        assert!(!admitted("Hello", 4, false, false));
        assert!(!admitted("", 4, false, false) && admitted("", 4, true, false));
        assert!(!admitted(" padded", 20, false, false));
        assert!(
            !admitted("line\nbreak", 20, false, false) && admitted("line\nbreak", 20, false, true)
        );
        assert!(!admitted("e\u{301}", 20, false, false), "not NFC");
        assert!(
            admitted("zero\u{200b}width", 20, false, true)
                && !admitted("zero\u{200b}width", 20, false, false),
            "a format character passes only beside layout, as in Python"
        );
        assert!(printable_with_layout("a\tb\nc") && !printable_with_layout("a\rb"));
        assert_eq!(hex(&[0, 171, 255]), "00abff");
    }
}
