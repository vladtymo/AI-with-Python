from fastmcp import FastMCP
from db import get_connection

mcp = FastMCP("Products MCP Server")

# MCP Tool:
# get_categories()
@mcp.tool()
def get_categories():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("""
        SELECT Id, Name
        FROM Categories
    """)

    rows = cursor.fetchall()

    categories = []

    for row in rows:

        categories.append({
            "Id": row.Id,
            "Name": row.Name
        })

    return categories


# MCP Tool:
# get_products_by_category(category_id)
@mcp.tool()
def get_products_by_category(category_id: int):

    connection = get_connection()

    cursor = connection.cursor()

    query = """
        SELECT
            p.Id,
            p.Title,
            p.Price,
            c.Name as Category
        FROM Products p
        JOIN Categories c
            ON p.CategoryId = c.Id
        WHERE c.Id = ?
        ORDER BY p.Price DESC
    """

    cursor.execute(query, category_id)

    rows = cursor.fetchall()

    products = []

    for row in rows:

        products.append({
            "Id": row.Id,
            "Title": row.Title,
            "Price": float(row.Price),
            "Category": row.Category
        })

    return products


# MCP Tool:
# get_top_products(category_id, top_n)
@mcp.tool()
def get_top_products(category_id: int, top_n: int = 3):

    connection = get_connection()

    cursor = connection.cursor()

    query = f"""
        SELECT TOP ({top_n})
            p.Id,
            p.Title,
            p.Price,
            c.Name as Category
        FROM Products p
        JOIN Categories c
            ON p.CategoryId = c.Id
        WHERE c.Id = ?
        ORDER BY p.Price DESC
    """

    cursor.execute(query, category_id)

    rows = cursor.fetchall()

    products = []

    for row in rows:

        products.append({
            "Id": row.Id,
            "Title": row.Title,
            "Price": float(row.Price),
            "Category": row.Category
        })

    return products


if __name__ == "__main__":

    mcp.run()