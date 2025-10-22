from typing import Dict

def split_resume_into_sections(resume_text: str) -> Dict[str, str]:
    """Split resume into sections based on markdown headers."""
    sections = {}
    current_section = "header"
    current_content = []

    for line in resume_text.split('\n'):
        if line.startswith('##'):
            # Save previous section
            if current_content:
                sections[current_section] = '\n'.join(current_content)
            # Start new section
            current_section = line.strip('# ').lower()
            current_content = [line]
        else:
            current_content.append(line)

    # Save last section
    if current_content:
        sections[current_section] = '\n'.join(current_content)

    return sections


def smart_truncate_resume(sections: Dict[str, str], max_chars: int) -> str:
    """Intelligently truncate resume, prioritizing important sections."""

    # Priority order for resume sections
    priority_sections = [
        'education', 'experience', 'work experience', 'skills',
        'projects', 'summary','publications', 'certifications','about',
        'achievements', 'awards'
    ]

    result = []
    current_length = 0

    # Add sections by priority
    for section_name in priority_sections:
        # Find matching section (case-insensitive partial match)
        for key, content in sections.items():
            if section_name in key.lower():
                section_length = len(content)
                if current_length + section_length <= max_chars:
                    result.append(content)
                    current_length += section_length
                elif current_length < max_chars:
                    # Add partial content
                    remaining = max_chars - current_length
                    result.append(content[:remaining] + "\n...[truncated]")
                    current_length = max_chars
                break

        if current_length >= max_chars:
            break

    # Add any remaining important sections not in priority list
    for key, content in sections.items():
        if current_length >= max_chars:
            break
        if key not in [s for s in priority_sections]:
            section_length = len(content)
            if current_length + section_length <= max_chars:
                result.append(content)
                current_length += section_length

    return '\n\n'.join(result)

