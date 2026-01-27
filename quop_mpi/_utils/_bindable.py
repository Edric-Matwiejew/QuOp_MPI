"""Bindable attributes functionality for QuOp Functions.

This module provides the Bindable class which enables automatic parameter
binding between QuOp Functions and class attributes.
"""

from __future__ import annotations


class Bindable:
    """Base class providing bindable attribute discovery for QuOp Functions.
    
    QuOp Functions can have their positional parameters automatically bound
    to class attributes by matching parameter names. This class provides
    methods to discover and display which attributes are available for binding.
    
    Subclasses should define a ``BINDABLE_ATTRIBUTES`` class variable as a
    dictionary mapping attribute names to description strings. The discovery
    methods automatically collect attributes from the entire class hierarchy,
    so subclasses can extend (not replace) the available bindings.
    
    Example
    -------
    .. code-block:: python
    
        class MyClass(Bindable):
            BINDABLE_ATTRIBUTES = {
                "my_attr": "Description of my_attr",
            }
            
            def __init__(self):
                self.my_attr = 42
        
        obj = MyClass()
        obj.print_bindable_attributes()
    
    See Also
    --------
    :term:`QuOp Function` : How parameter binding works
    """
    
    # Subclasses override this to define their bindable attributes
    BINDABLE_ATTRIBUTES: dict[str, str] = {}

    @classmethod
    def _collect_bindable_attributes(cls) -> dict[str, str]:
        """Collect BINDABLE_ATTRIBUTES from the entire class hierarchy.
        
        This allows subclasses to extend the bindable attributes by defining
        their own BINDABLE_ATTRIBUTES dict. Subclass definitions override
        parent definitions for the same key.
        
        Returns
        -------
        dict
            Merged dictionary of all bindable attributes from the class hierarchy.
        """
        result = {}
        # Traverse MRO in reverse so subclass definitions override parents
        for klass in reversed(cls.__mro__):
            attrs = getattr(klass, 'BINDABLE_ATTRIBUTES', None)
            if attrs is not None:
                result.update(attrs)
        return result

    def get_bindable_attributes(self) -> dict[str, tuple]:
        """Return a dictionary of attributes available for binding to QuOp Functions.

        QuOp Functions can have their positional parameters automatically bound
        to class attributes by matching parameter names. This method shows which
        attributes are available for binding and their current values.
        
        This method collects bindable attributes from the entire class hierarchy,
        so subclasses can extend the available attributes.

        Returns
        -------
        dict
            Dictionary mapping attribute names to (value, description) tuples.
            Value is None if the attribute hasn't been set yet.

        Examples
        --------
        >>> obj.get_bindable_attributes()
        {'system_size': (1024, 'Total number of quantum basis states'),
         'local_i': (None, 'Number of elements in this rank\\'s partition'),
         ...}

        See Also
        --------
        print_bindable_attributes : Print formatted table
        """
        all_attrs = self._collect_bindable_attributes()
        result = {}
        for attr, description in all_attrs.items():
            value = getattr(self, attr, None)
            result[attr] = (value, description)
        return result

    def print_bindable_attributes(self):
        """Print a formatted table of attributes available for binding to QuOp Functions.

        This is a convenience method for interactive use to discover which
        parameter names can be used in custom QuOp Functions.
        """
        all_attrs = self._collect_bindable_attributes()
        class_name = self.__class__.__name__
        
        # Include unitary_type if available (for Unitary subclasses)
        unitary_type = getattr(self, 'unitary_type', None)
        if unitary_type:
            header = f"\nBindable Attributes for {class_name} ({unitary_type})"
        else:
            header = f"\nBindable Attributes for {class_name}"
        
        print(header)
        print("=" * 70)
        print(f"{'Attribute':<25} {'Set?':<6} Description")
        print("-" * 70)
        for attr, description in all_attrs.items():
            value = getattr(self, attr, None)
            is_set = "Yes" if value is not None else "No"
            print(f"{attr:<25} {is_set:<6} {description}")
        print()
