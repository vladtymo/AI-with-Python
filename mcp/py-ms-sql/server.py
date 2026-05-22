from flask import Flask, jsonify
from db import get_connection

app = Flask(__name__)

"""
MCP Tool:
get_products(category)
"""
@app.route("/products/<category>", methods=["GET"])
def get_products(category):

    try:

        connection = get_connection()

        cursor = connection.cursor()

        query = """
            SELECT *
            FROM Products
            WHERE Category = ?
            ORDER BY Price DESC
        """

        cursor.execute(query, category)

        rows = cursor.fetchall()

        products = []

        for row in rows:

            products.append({
                "Id": row.Id,
                "Name": row.Name,
                "Category": row.Category,
                "Price": float(row.Price)
            })

        return jsonify({
            "tool": "get_products",
            "category": category,
            "count": len(products),
            "products": products
        })

    except Exception as error:

        return jsonify({
            "error": str(error)
        }), 500


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=3000,
        debug=True
    )