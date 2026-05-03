from django import template

register = template.Library()

@register.filter
def slugify_class(value):
    """Convert class name like 'storage-tank' to CSS-safe slug."""
    return value.replace(" ", "-").lower()
